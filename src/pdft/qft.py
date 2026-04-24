"""Quantum Fourier Transform circuit as a hand-rolled tensor network.

Mirror of upstream src/qft.jl (which calls Yao.EasyBuild.qft_circuit +
yao2einsum). We replicate Yao's compact tensor representation:

- Hadamard on qubit q: 2x2 tensor `H[out, in]` that transforms the wire
  label (introducing a fresh output label).
- Controlled-phase on qubits (c, t) with angle phi: 2x2 DIAGONAL tensor
  `CP[c_wire, t_wire] = diag of the 4x4 gate`, i.e.

        CP[0,0]=1, CP[0,1]=1, CP[1,0]=1, CP[1,1]=exp(i*phi)

  The CP shares the CURRENT wire labels of its control and target qubits
  and does NOT introduce new labels — the gate is diagonal so each wire's
  state is unchanged in value; only a multiplicative phase is injected.

This compact form (a) matches Yao's output exactly so tensor values align
with Julia goldens element-wise, and (b) uses fewer tensor legs, which
makes the einsum contraction cheaper.

Gate sequence (per upstream src/entangled_qft.jl:51-77):

    For j = 1..n:
        H on qubit j
        For target in j+1..n:
            CP(control=target, target=j, phase=2*pi/2^(target-j+1))

2D QFT = (m-qubit QFT on row qubits) tensor (n-qubit QFT on col qubits);
no entanglement between blocks.

Endianness: qubit 1 maps to the LOWEST reshape axis of pic (little-endian,
matching Yao's convention). For a 2^m x 2^n image reshaped to
(2,)*(m+n), the legs correspond (in order) to row qubits 1..m then column
qubits 1..n, with the lowest-indexed qubit on the leftmost axis.
"""
from __future__ import annotations

import string
from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp

from .einsum_cache import optimize_code_cached

Array = jax.Array


HADAMARD: Array = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)


def controlled_phase_diag(phi: float) -> Array:
    """2x2 diagonal of a 4x4 controlled-phase gate.

    Indexed as `T[c, t]` (control, target): the value is 1 except when
    both c=1 AND t=1, in which case it is exp(i*phi). This is the compact
    tensor-network representation Yao's `yao2einsum` emits.
    """
    return jnp.array(
        [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, jnp.exp(1j * phi)]],
        dtype=jnp.complex128,
    )


class _Gate(TypedDict):
    kind: str  # "H" or "CP"
    qubits: tuple[int, ...]
    tensor: Array
    phase: float


def _qft_gates_1d(n_qubits: int, offset: int) -> list[_Gate]:
    """Emit the 1D QFT gate sequence on qubits (offset+1, ..., offset+n_qubits).

    Matches upstream src/entangled_qft.jl:64-78 exactly.
    """
    gates: list[_Gate] = []
    for j in range(1, n_qubits + 1):
        q = offset + j
        gates.append(_Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))
        for target in range(j + 1, n_qubits + 1):
            k = target - j + 1
            t = offset + target
            phi = 2 * jnp.pi / (2 ** k)
            gates.append(
                _Gate(
                    kind="CP",
                    qubits=(t, q),  # (control, target) — matches upstream convention
                    tensor=controlled_phase_diag(float(phi)),
                    phase=float(phi),
                )
            )
    return gates


# ---------------------------------------------------------------------------
# Einsum subscript builder
# ---------------------------------------------------------------------------


def _build_qft_einsum(m: int, n: int, *, inverse: bool):
    """Construct subscripts, tensors (in gate order), operand shapes.

    Returns (subscripts, tensors, operand_shapes) where:
        - subscripts: "ijkl,...,<pic_legs>-><out_legs>"
        - tensors: list of Array in the order they appear in subscripts
        - operand_shapes: list of tuples, in operand order (tensors first,
          then pic) — used by einsum_cache.optimize_code_cached.

    Tensor shapes and subscripts match Julia's compact representation: H is
    (2,2) with legs (out, in); CP is (2,2) with legs (control_wire, target_wire).
    """
    N = m + n
    pool = list(string.ascii_lowercase + string.ascii_uppercase)
    next_idx = 0

    def fresh() -> str:
        nonlocal next_idx
        if next_idx >= len(pool):
            raise ValueError(f"too many qubits: need > {len(pool)} einsum labels")
        ch = pool[next_idx]
        next_idx += 1
        return ch

    input_labels = [fresh() for _ in range(N)]
    wire_state: dict[int, str] = {q + 1: input_labels[q] for q in range(N)}

    gates = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)

    tensor_subscripts: list[str] = []
    tensor_list: list[Array] = []
    tensor_shapes: list[tuple[int, ...]] = []

    for g in gates:
        if g["kind"] == "H":
            (q,) = g["qubits"]
            in_lbl = wire_state[q]
            out_lbl = fresh()
            tensor_subscripts.append(out_lbl + in_lbl)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2))
            wire_state[q] = out_lbl
        elif g["kind"] == "CP":
            q_ctrl, q_tgt = g["qubits"]
            # Diagonal CP reuses the current wire labels — no new labels.
            ctrl_lbl = wire_state[q_ctrl]
            tgt_lbl = wire_state[q_tgt]
            tensor_subscripts.append(ctrl_lbl + tgt_lbl)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2))
        else:
            raise AssertionError(f"unknown gate kind: {g['kind']}")

    # Match Julia's `perm_vec = sortperm(tn.tensors, by=x -> !(x ≈ mat(H)))`:
    # reorder tensors + their subscripts so Hadamards come first, then
    # diagonal CPs. The einsum remains valid because we permute tensor_list
    # and tensor_subscripts with the same permutation. Required for
    # cross-language JSON interop (Phase 2) — Julia's serialization uses
    # this order.
    import numpy as _np

    H_np = _np.asarray(HADAMARD)
    is_not_hadamard = [
        not _np.allclose(_np.asarray(t), H_np, atol=1e-12) for t in tensor_list
    ]
    perm = sorted(range(len(tensor_list)), key=lambda i: is_not_hadamard[i])
    tensor_list = [tensor_list[i] for i in perm]
    tensor_subscripts = [tensor_subscripts[i] for i in perm]
    # tensor_shapes at this point has exactly len(tensor_list) entries (pic
    # shape is appended below, after this block). Permute in place.
    tensor_shapes = [tensor_shapes[i] for i in perm]

    # Yao uses little-endian qubit ordering: qubit 1 = LSB of the state index.
    # Python's `pic.reshape((2,)*(m+n))` gives axes in big-endian order
    # (axis 0 = MSB of row, ..., axis m-1 = LSB of row, then same for cols).
    # To make axis i ↔ qubit (m-i) within the row block and axis (m+k) ↔
    # qubit (m+n-k) within the col block, we assign pic/out labels in
    # within-block reversed qubit order.
    row_pic = [input_labels[q - 1] for q in range(m, 0, -1)]          # axes 0..m-1 → qubits m..1
    col_pic = [input_labels[q - 1] for q in range(m + n, m, -1)]      # axes m..m+n-1 → qubits m+n..m+1
    pic_labels = "".join(row_pic + col_pic)

    row_out = [wire_state[q] for q in range(m, 0, -1)]
    col_out = [wire_state[q] for q in range(m + n, m, -1)]
    out_labels = "".join(row_out + col_out)

    if inverse:
        lhs = ",".join(tensor_subscripts + [out_labels])
        rhs = pic_labels
    else:
        lhs = ",".join(tensor_subscripts + [pic_labels])
        rhs = out_labels

    subscripts = f"{lhs}->{rhs}"
    tensor_shapes.append((2,) * N)
    return subscripts, tensor_list, tensor_shapes


def qft_code(m: int, n: int, *, inverse: bool = False) -> tuple[Callable[..., Array], list[Array]]:
    """Return `(einsum_fn, initial_tensors)` for 2D QFT on (2^m, 2^n) images.

    Mirror of upstream src/qft.jl:20-39. The `code` callable expects
    `len(initial_tensors)` tensor operands followed by the reshaped image.
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    subscripts, tensors, shapes = _build_qft_einsum(m, n, inverse=inverse)
    code = optimize_code_cached(subscripts, *shapes)
    return code, tensors


# ---------------------------------------------------------------------------
# Forward / inverse transforms
# ---------------------------------------------------------------------------


def ft_mat(tensors: list[Array], code: Callable, m: int, n: int, pic: Array) -> Array:
    """Apply 2D QFT circuit to a (2^m, 2^n) image.

    Mirror of upstream src/qft.jl:60-63.
    """
    if pic.shape != (2 ** m, 2 ** n):
        raise ValueError(
            f"pic shape must be (2**m, 2**n) = ({2**m}, {2**n}), got {pic.shape}"
        )
    reshaped = pic.astype(jnp.complex128).reshape((2,) * (m + n))
    out = code(*tensors, reshaped)
    return out.reshape(2 ** m, 2 ** n)


def ift_mat(tensors: list[Array], code: Callable, m: int, n: int, pic: Array) -> Array:
    """Apply 2D inverse QFT circuit.

    Mirror of upstream src/qft.jl:80-83. Note: caller is expected to have
    already conjugated the tensors (that mirrors upstream's convention of
    combining `conj.(tensors)` with the inverse einsum code).
    """
    if pic.shape != (2 ** m, 2 ** n):
        raise ValueError(
            f"pic shape must be (2**m, 2**n) = ({2**m}, {2**n}), got {pic.shape}"
        )
    reshaped = pic.astype(jnp.complex128).reshape((2,) * (m + n))
    out = code(*tensors, reshaped)
    return out.reshape(2 ** m, 2 ** n)
