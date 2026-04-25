"""Shared circuit-to-einsum builder used by qft, entangled_qft, tebd, mera.

Each concrete circuit family produces a list of gates (H and compact CP) via
its own `_gates_...` function, then calls `build_circuit_einsum` to turn
them into an `(einsum_fn, tensors)` pair with the conventions that match
Julia's Yao + yao2einsum output:

- Hadamard tensor is shared (HADAMARD).
- Controlled-phase tensors are 2x2 diagonal `[[1, 1], [1, exp(i*phi)]]`
  (Yao's compact tensor network form), *not* 4x4 CP matrices.
- Tensors are sorted Hadamards-first (matches Julia's `perm_vec`).
- pic and output leg order uses Yao's little-endian convention: qubit 1
  maps to the LOWEST-index reshape axis within each block, hence we
  reverse within-block qubit order for both pic_labels and out_labels.
"""

from __future__ import annotations

import string
from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp

Array = jax.Array


HADAMARD: Array = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / jnp.sqrt(2.0)


def controlled_phase_diag(phi: float) -> Array:
    """2x2 compact representation of a 4x4 controlled-phase gate.

    `T[c, t] = 1` except `T[1, 1] = exp(i*phi)`. Yao emits this form for
    diagonal controlled-phase gates; we match it so tensor values align
    with Julia goldens element-wise.
    """
    return jnp.array(
        [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, jnp.exp(1j * phi)]],
        dtype=jnp.complex128,
    )


class Gate(TypedDict):
    kind: str  # "H" or "CP"
    qubits: tuple[int, ...]
    tensor: Array
    phase: float


def build_circuit_einsum(
    gates: list[Gate],
    m: int,
    n: int,
    *,
    inverse: bool,
) -> tuple[str, list[Array], list[tuple[int, ...]]]:
    """Convert a gate sequence to (subscripts, tensors, shapes).

    `gates` is applied in order to a circuit on `m + n` qubits (1-indexed).
    For CP gates, the 2x2 tensor shares the current wire labels of its
    control and target qubits and does NOT introduce new labels (the gate
    is diagonal).

    Returns the triple needed by `einsum_cache.optimize_code_cached`.
    The returned tensor list and tensor-shape list are sorted so Hadamards
    come first (matching Julia's `perm_vec`).
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
            ctrl_lbl = wire_state[q_ctrl]
            tgt_lbl = wire_state[q_tgt]
            tensor_subscripts.append(ctrl_lbl + tgt_lbl)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2))
        else:
            raise AssertionError(f"unknown gate kind: {g['kind']}")

    # Hadamard-first sort
    import numpy as _np

    H_np = _np.asarray(HADAMARD)
    is_not_hadamard = [not _np.allclose(_np.asarray(t), H_np, atol=1e-12) for t in tensor_list]
    perm = sorted(range(len(tensor_list)), key=lambda i: is_not_hadamard[i])
    tensor_list = [tensor_list[i] for i in perm]
    tensor_subscripts = [tensor_subscripts[i] for i in perm]
    tensor_shapes = [tensor_shapes[i] for i in perm]

    # Little-endian qubit mapping within each block
    row_pic = [input_labels[q - 1] for q in range(m, 0, -1)]
    col_pic = [input_labels[q - 1] for q in range(m + n, m, -1)]
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
    tensor_shapes.append((2,) * N)  # pic operand
    return subscripts, tensor_list, tensor_shapes


def compile_circuit(
    gates: list[Gate],
    m: int,
    n: int,
    *,
    inverse: bool,
) -> tuple[Callable[..., Array], list[Array]]:
    """High-level entry: build subscripts and jit-compile the einsum."""
    from .einsum_cache import optimize_code_cached

    subscripts, tensors, shapes = build_circuit_einsum(gates, m, n, inverse=inverse)
    code = optimize_code_cached(subscripts, *shapes)
    return code, tensors


def apply_circuit(
    tensors: list[Array],
    code: Callable,
    m: int,
    n: int,
    pic: Array,
) -> Array:
    """Contract pic through the circuit and reshape back to (2^m, 2^n)."""
    if pic.shape != (2**m, 2**n):
        raise ValueError(f"pic shape must be (2**m, 2**n) = ({2**m}, {2**n}), got {pic.shape}")
    reshaped = pic.astype(jnp.complex128).reshape((2,) * (m + n))
    out = code(*tensors, reshaped)
    return out.reshape(2**m, 2**n)


# ---------------------------------------------------------------------------
# Phase extraction helpers (shared across entangled_qft / tebd / mera)
# ---------------------------------------------------------------------------


def is_compact_cp(tensor: Array, atol: float = 0.15) -> bool:
    """True if `tensor` looks like a compact 2x2 CP gate `[[1, 1], [1, e^iφ]]`.

    All four entries should have unit magnitude (within `atol`). Mirror of
    upstream src/tebd.jl:124-132 / src/mera.jl:190-198.
    """
    import numpy as np

    arr = np.asarray(tensor)
    if arr.shape != (2, 2):
        return False
    return all(abs(abs(arr[i, j]) - 1.0) <= atol for i in range(2) for j in range(2))


def extract_phase_from_cp(tensor: Array) -> float:
    """Extract `φ` from a compact 2x2 CP tensor `[[1, 1], [1, e^iφ]]`.

    Mirror of upstream `extract_*_phases` (uses `angle(tensors[idx][2, 2])`
    in Julia 1-based indexing = `tensor[1, 1]` in Python).
    """
    import numpy as np

    arr = np.asarray(tensor)
    return float(np.angle(arr[1, 1]))


def select_last_n_cp_indices(tensors: list[Array], n_gates: int) -> list[int]:
    """Return indices of the LAST `n_gates` compact-CP tensors in `tensors`.

    Mirror of upstream `get_*_gate_indices`. Sorts by position in the list,
    then takes the last `n_gates` such indices. After training tensors may
    drift slightly from the exact pattern; the unit-modulus check uses a
    moderate tolerance to absorb that.
    """
    cp_indices = [i for i, t in enumerate(tensors) if is_compact_cp(t)]
    if len(cp_indices) >= n_gates:
        return cp_indices[-n_gates:]
    return cp_indices
