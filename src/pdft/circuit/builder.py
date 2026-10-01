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
from dataclasses import dataclass
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
    """One gate of a circuit program.

    ``kind`` is ``"H"``, ``"CP"``, ``"U4"`` or ``"CRY"``; ``qubits`` are the
    wires it acts on; ``tensor`` is its einsum tensor; ``phase`` is the angle
    of a CP or CRY gate.
    """

    kind: str  # "H", "CP", "U4", or "CRY"
    qubits: tuple[int, ...]
    tensor: Array
    phase: float


def u4_from_phase(phi: float) -> Array:
    """Return the 2-qubit (2, 2, 2, 2)-shaped tensor matching the 4×4 controlled-
    phase gate diag(1, 1, 1, exp(iφ)).

    This is the strict warm-start for a learnable U(4) gate that should
    behave initially like a fixed CP. Axis order: (out_ctrl, out_tgt,
    in_ctrl, in_tgt).
    """
    diag = jnp.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j, jnp.exp(1j * phi)], dtype=jnp.complex128)
    return jnp.diag(diag).reshape(2, 2, 2, 2)


def hadamard_gate(q: int) -> Gate:
    """A Hadamard on qubit ``q``."""
    return Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0)


_PHASE_FORMS = {"cp": ("CP", controlled_phase_diag), "u4": ("U4", u4_from_phase)}


def phase_gate(parametrization: str) -> Callable[[int, int, float], Gate]:
    """The constructor ``gate(q_ctrl, q_tgt, phi)`` of a controlled phase in one of its two forms.

    ``"cp"`` stores the gate as the compact diagonal tensor, trained on
    ``U(1)^4``. ``"u4"`` stores the same operator as a dense two-qubit tensor,
    trained on ``U(4)``: both start equal for a given ``phi`` and differ only
    in the manifold they relax over.
    """
    if parametrization not in _PHASE_FORMS:
        raise ValueError(f"parametrization must be 'cp' or 'u4', got {parametrization!r}")
    kind, tensor = _PHASE_FORMS[parametrization]

    def gate(q_ctrl: int, q_tgt: int, phi: float) -> Gate:
        return Gate(kind=kind, qubits=(q_ctrl, q_tgt), tensor=tensor(phi), phase=phi)

    return gate


cp_gate = phase_gate("cp")
u4_gate = phase_gate("u4")


def two_registers(emit_1d: Callable[[int, int], list[Gate]], m: int, n: int) -> list[Gate]:
    """A separable 2-D circuit: ``emit_1d(n_qubits, offset)`` on the row qubits, then on the column qubits.

    Rows are qubits ``1..m`` and columns ``m+1..m+n``; nothing couples the two registers.
    """
    check_qubits(m, n)
    return emit_1d(m, 0) + emit_1d(n, m)


def check_qubits(m: int, n: int) -> None:
    """Every 2-D circuit needs at least one qubit per register."""
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")


def _hadamard_first_perm(tensor_list: list[Array]) -> list[int]:
    """Indices that sort `tensor_list` Hadamards-first (stable), matching
    Julia's `perm_vec`. Non-Hadamard tensors keep their relative order."""
    import numpy as _np

    H_np = _np.asarray(HADAMARD)

    def _is_hadamard(t):
        a = _np.asarray(t)
        return a.shape == (2, 2) and _np.allclose(a, H_np, atol=1e-12)

    is_not_hadamard = [not _is_hadamard(t) for t in tensor_list]
    return sorted(range(len(tensor_list)), key=lambda i: is_not_hadamard[i])


@dataclass(frozen=True)
class Program:
    """The structure of a circuit: which gate acts on which qubits, in what order.

    It holds no tensor values, so it is hashable and can be a static argument
    of a jitted function: every basis with the same program shares one
    compiled applier. ``steps`` is the gate sequence in temporal order.
    ``slot[i]`` is the position of step ``i``'s tensor in the tensor list a
    basis stores, which is sorted Hadamards-first to match Julia's
    ``perm_vec``; the sort is decided once, from the initial tensor values,
    by ``compile_program``.
    """

    m: int
    n: int
    steps: tuple[tuple[str, tuple[int, ...]], ...]
    slot: tuple[int, ...]

    @property
    def sorted_steps(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """``(kind, qubits)`` of each tensor, in the order the tensor list stores them."""
        by_slot = dict(zip(self.slot, self.steps))
        return tuple(by_slot[i] for i in range(len(self.steps)))


def compile_program(gates: list[Gate], m: int, n: int) -> tuple[Program, list[Array]]:
    """The program of a gate sequence on ``m + n`` qubits, and its tensors in stored order."""
    perm = _hadamard_first_perm([g["tensor"] for g in gates])
    slot = [0] * len(gates)
    for position, step in enumerate(perm):
        slot[step] = position
    steps = tuple((g["kind"], tuple(g["qubits"])) for g in gates)
    return Program(m, n, steps, tuple(slot)), [gates[step]["tensor"] for step in perm]


def sorted_gate_program(gates: list[Gate]) -> list[tuple[str, tuple[int, ...]]]:
    """Return `(kind, qubits)` per tensor in sorted operand order.

    Uses the same Hadamard-first sort `compile_circuit` applies, so the
    returned order lines up element-for-element with the tensor list a basis
    stores. Handy for mapping gates (and the qubits they touch) to tensor
    indices after compilation.
    """
    perm = _hadamard_first_perm([g["tensor"] for g in gates])
    return [(gates[p]["kind"], gates[p]["qubits"]) for p in perm]


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
        elif g["kind"] == "U4":
            # General 2-qubit unitary stored as (out_c, out_t, in_c, in_t).
            # Unlike CP this is NOT diagonal; it INTRODUCES new wire labels
            # for both qubits' outputs.
            q_ctrl, q_tgt = g["qubits"]
            in_c = wire_state[q_ctrl]
            in_t = wire_state[q_tgt]
            out_c = fresh()
            out_t = fresh()
            tensor_subscripts.append(out_c + out_t + in_c + in_t)
            tensor_list.append(g["tensor"])
            tensor_shapes.append((2, 2, 2, 2))
            wire_state[q_ctrl] = out_c
            wire_state[q_tgt] = out_t
        else:
            # NOTE: "CRY" is handled only by the stepped path (_walk);
            # the legacy einsum builder does not emit it.
            raise AssertionError(f"unknown gate kind: {g['kind']}")

    # Hadamard-first sort (matches Julia's perm_vec).
    perm = _hadamard_first_perm(tensor_list)
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


def _axis_of_qubit(q: int, m: int, n: int) -> int:
    """Map a 1-indexed qubit number to its position in the (2,)*N reshape of pic.

    Yao's little-endian convention (matching `build_circuit_einsum`'s
    `pic_labels` ordering): qubit `m` maps to axis 0, qubit `m-1` to axis 1,
    …, qubit 1 to axis m-1, qubit `m+n` to axis m, …, qubit `m+1` to axis
    m+n-1.
    """
    if 1 <= q <= m:
        return m - q
    if m + 1 <= q <= m + n:
        return m + (m + n - q)
    raise ValueError(f"qubit index {q} out of range (1..{m + n})")


def _one_qubit(pic: Array, ax: int, T: Array, inverse: bool, slices: bool) -> Array:
    """Apply the one-qubit gate ``T[out, in]`` to axis ``ax`` of ``pic``.

    ``inverse`` applies the transpose of ``T``. Combined with the caller's
    ``conj(T)`` this is the true adjoint ``conj(T.T)``; for a symmetric
    Hadamard the leg swap changes nothing, but for a trained, non-symmetric
    gate the round trip fails without it.

    Two arithmetics compute the same thing. The default contracts ``T`` with
    the axis, asking for exact matmul precision: in single precision XLA
    otherwise lowers the contraction to TF32 on recent GPUs, which costs three
    digits. ``slices`` combines the two slices of the axis explicitly, with no
    contraction at all. Measured on an L1 gradient of ``QFTBasis``: on a GPU
    slices are 2 to 6 times faster at every size (41 ms against 7 ms at
    512x512); on a CPU they are up to 1.7 times slower below 256x256 and
    compile twice as slowly. Hence opt-in.
    """
    if slices:
        U = T.T if inverse else T
        a, b = jnp.take(pic, 0, axis=ax), jnp.take(pic, 1, axis=ax)
        return jnp.stack([U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b], axis=ax)
    out = jnp.tensordot(T, pic, axes=[[0 if inverse else 1], [ax]], precision="highest")
    return jnp.moveaxis(out, 0, ax)


def _walk(program: Program, inverse: bool, slices: bool, tensors: tuple, pic: Array) -> Array:
    """Apply the program to ``pic``, shape ``(..., *(2,) * (m + n))``, one gate at a time.

    Leading axes of ``pic`` are batch axes. ``tensors`` is in stored
    (Hadamard-first) order. ``inverse`` walks the steps backwards with each
    gate's legs swapped, which is the transpose of the circuit; the caller
    conjugates the tensors to make it the adjoint, as Julia's
    ``inverse_code(conj.(tensors)...)`` does. ``slices`` selects the arithmetic
    of the one-qubit gates, see ``_one_qubit``.
    """
    m, n = program.m, program.n
    lead = pic.ndim - (m + n)
    order = range(len(program.steps))
    for i in reversed(order) if inverse else order:
        kind, qubits = program.steps[i]
        T = tensors[program.slot[i]]
        axes = [lead + _axis_of_qubit(q, m, n) for q in qubits]
        if kind == "H":
            pic = _one_qubit(pic, axes[0], T, inverse, slices)
        elif kind == "CP":
            # T[c, t] is the diagonal of the gate, broadcast onto its two wires.
            # A reshape puts T's first axis on the lower-numbered axis, so T is
            # transposed when the control sits on the higher one.
            ax_c, ax_t = axes
            shape = [1] * pic.ndim
            shape[ax_c] = shape[ax_t] = 2
            pic = pic * (T if ax_c < ax_t else T.T).reshape(shape)
        elif kind == "U4":
            # T[oc, ot, ic, it]: forward contracts the input legs, inverse the
            # output legs (the transpose; the caller conjugates).
            legs = [0, 1] if inverse else [2, 3]
            pic = jnp.tensordot(T, pic, axes=[legs, axes], precision="highest")
            pic = jnp.moveaxis(pic, [0, 1], axes)
        elif kind == "CRY":
            # A (2, 2) block applied to the target where the control is 1 and
            # nothing where it is 0. Slicing the control axis touches half the
            # amplitudes, instead of rotating everything and masking half away.
            ax_c, ax_t = axes
            passed = jnp.take(pic, 0, axis=ax_c)
            rotated = jnp.take(pic, 1, axis=ax_c)
            # the target's axis once the control axis is sliced out
            rotated = _one_qubit(rotated, ax_t if ax_t < ax_c else ax_t - 1, T, inverse, slices)
            pic = jnp.stack([passed, rotated], axis=ax_c)
        else:
            raise AssertionError(f"unknown gate kind: {kind}")
    return pic


_run = jax.jit(_walk, static_argnames=("program", "inverse", "slices"))


@dataclass(frozen=True)
class CircuitCode:
    """``code(*tensors, pic)``: a program applied to ``pic`` of shape ``(..., *(2,) * (m + n))``.

    The callable a basis keeps as ``code`` and ``inv_code``. It compares and
    hashes by its program, so two bases with the same circuit share one
    compiled applier and have equal pytree structures. ``slices`` selects the
    arithmetic of the one-qubit gates, see ``_one_qubit``.
    """

    program: Program
    inverse: bool = False
    slices: bool = False

    def __call__(self, *operands: Array) -> Array:
        *tensors, pic = operands
        return _run(self.program, self.inverse, self.slices, tuple(tensors), pic)


def apply_program(
    program: Program, tensors, x: Array, *, inverse: bool = False, slices: bool = False
) -> Array:
    """Apply a program to an image, or to a stack of images.

    ``x`` has shape ``(..., 2**m, 2**n)``; leading axes are batch axes. The
    working precision follows the image: complex64 for a float32 or complex64
    image, complex128 otherwise, and the tensors are cast to it. For the
    adjoint pass conjugated tensors with ``inverse=True``. ``slices`` selects
    the arithmetic of the one-qubit gates, faster on a GPU and slower on a CPU
    at small sizes; see ``_one_qubit``.

    ``apply_circuit`` is the fixed-precision, single-image entry point every
    basis has always used; this is the same walk without those two limits.
    """
    m, n = program.m, program.n
    if x.shape[-2:] != (2**m, 2**n):
        raise ValueError(f"image shape must end in ({2**m}, {2**n}), got {x.shape}")
    dtype = jnp.complex64 if x.dtype in (jnp.float32, jnp.complex64) else jnp.complex128
    pic = x.astype(dtype).reshape(x.shape[:-2] + (2,) * (m + n))
    out = _run(program, inverse, slices, tuple(t.astype(dtype) for t in tensors), pic)
    return out.reshape(x.shape)


def compile_circuit(
    gates: list[Gate],
    m: int,
    n: int,
    *,
    inverse: bool,
) -> tuple[CircuitCode, list[Array]]:
    """The applier of a gate sequence and its initial tensors.

    Returns ``(code, tensors)`` where ``code(*tensors, pic_reshaped)`` applies
    the gates one at a time, and ``tensors`` is in the Hadamard-first order
    saved checkpoints serialise.
    """
    program, tensors = compile_program(gates, m, n)
    return CircuitCode(program, inverse), tensors


def apply_circuit(
    tensors: list[Array],
    code: CircuitCode,
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


def extract_phases(tensors: list[Array], indices: list[int]) -> list[float]:
    """The phase of each compact CP tensor at ``indices``, in that order."""
    return [extract_phase_from_cp(tensors[idx]) for idx in indices]


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
