"""Gates, programs and the one applier every circuit basis runs on.

A circuit family emits a list of ``Gate``; ``compile_program`` turns the list
into a ``Program`` (the structure, as hashable data) and the tensors in stored
order; ``CircuitCode`` applies a program to an image one gate at a time. The
conventions match Julia's Yao + ``yao2einsum`` output:

- The Hadamard tensor is shared (``HADAMARD``).
- A controlled phase is the compact 2x2 tensor ``[[1, 1], [1, exp(i*phi)]]``
  Yao emits for a diagonal gate, not a 4x4 matrix.
- Tensors are stored Hadamards-first (Julia's ``perm_vec``); a program's
  ``slot`` maps each gate to its stored tensor.
- Qubits are little-endian within each register: qubit 1 is the last reshape
  axis of its block (``_axis_of_qubit``).

Gates are applied one at a time instead of as one einsum, so a circuit is not
bounded by the 52 einsum labels and needs no contraction path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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


# The gate kinds, each with the shape of the tensor it stores. "H" is any
# one-qubit gate (it starts as a Hadamard or a rotation), "CP" a diagonal
# two-qubit gate in compact 2x2 form, "U4" a dense two-qubit gate and "CRY" a
# one-qubit block applied where the control is 1.
GATE_SHAPES = {"H": (2, 2), "CP": (2, 2), "U4": (2, 2, 2, 2), "CRY": (2, 2)}
GATE_KINDS = tuple(GATE_SHAPES)
REGISTERS = ("row", "column", "both")


class Gate(TypedDict):
    """One gate of a circuit program.

    ``kind`` is one of ``GATE_KINDS``; ``qubits`` are the wires it acts on;
    ``tensor`` is its tensor, of shape ``GATE_SHAPES[kind]``. ``phase`` is the
    angle the gate was built from, 0.0 when it has none. It is a record for
    the reader: what is applied is the tensor.
    """

    kind: str
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
    return controlled(jnp.diag(jnp.array([1.0 + 0j, jnp.exp(1j * phi)], dtype=jnp.complex128)))


def controlled(block: Array) -> Array:
    """The dense two-qubit tensor that applies the one-qubit ``block`` where the control is 1.

    ``block[out, in]`` acts on the target; where the control is 0 nothing
    happens. Axis order: (out_ctrl, out_tgt, in_ctrl, in_tgt), the storage of
    a ``"U4"`` gate. A controlled phase, a CNOT and a controlled rotation are
    this with a diagonal, a bit flip and a rotation for ``block``.
    """
    tensor = jnp.zeros((2, 2, 2, 2), dtype=jnp.complex128)
    tensor = tensor.at[0, :, 0, :].set(jnp.eye(2, dtype=jnp.complex128))
    return tensor.at[1, :, 1, :].set(block)


def identity_tensor(kind: str) -> Array:
    """The tensor with which a gate of ``kind`` does nothing, as complex128."""
    if kind == "CP":
        return controlled_phase_diag(0.0)  # all ones: the compact form of diag(1, 1, 1, 1)
    if kind == "U4":
        return controlled(jnp.eye(2, dtype=jnp.complex128))
    if kind in GATE_SHAPES:
        return jnp.eye(2, dtype=jnp.complex128)
    raise AssertionError(f"unknown gate kind: {kind}")


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


def phase_list(phases: Sequence[float] | None, count: int, requirement: str) -> list[float]:
    """``count`` angles as floats: zeros when ``phases`` is ``None``, else ``phases`` checked for length.

    ``requirement`` is the sentence a wrong length is reported with.
    """
    angles = [0.0] * count if phases is None else [float(p) for p in phases]
    if len(angles) != count:
        raise ValueError(f"{requirement}, got {len(angles)}")
    return angles


def hadamards_then_layers(
    layer: Callable[[int, int, list[float], Callable[[int, int, float], Gate]], list[Gate]],
    count: Callable[[int], int],
    m: int,
    n: int,
    phases: Sequence[float] | None,
    parametrization: str,
) -> tuple[list[Gate], int, int]:
    """A Hadamard on every qubit, then one layer of phase gates on each register.

    The shape TEBD and MERA share; they differ in the layer.
    ``count(n_qubits)`` is the number of gates in a register's layer and
    ``layer(n_qubits, offset, phases, gate)`` emits them, building each with
    ``gate(q_ctrl, q_tgt, phi)``. ``phases`` holds the row layer's angles and
    then the column layer's; ``None`` means all zero. Returns
    ``(gates, n_row_gates, n_col_gates)``.
    """
    check_qubits(m, n)
    gate = phase_gate(parametrization)
    n_row, n_col = count(m), count(n)
    angles = phase_list(
        phases,
        n_row + n_col,
        f"phases must have length {n_row + n_col} ({n_row} row + {n_col} column gates)",
    )
    gates = [hadamard_gate(q) for q in range(1, m + n + 1)]
    gates += layer(m, 0, angles[:n_row], gate) + layer(n, m, angles[n_row:], gate)
    return gates, n_row, n_col


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

    def tensor_indices(self, kind: str | None = None, register: str | None = None) -> list[int]:
        """Positions in the stored tensor list of the gates of one kind, on one register, or both.

        ``kind`` is one of ``GATE_KINDS``. ``register`` is ``"row"`` (every
        qubit of the gate is one of ``1..m``), ``"column"`` (``m+1..m+n``) or
        ``"both"`` (a gate that couples the two). The result is sorted, ready
        to be passed as ``frozen_indices`` or to index a parameter view:
        ``program.tensor_indices(kind="H")`` freezes every one-qubit gate.
        """
        if kind is not None and kind not in GATE_KINDS:
            raise ValueError(f"kind must be one of {GATE_KINDS}, got {kind!r}")
        if register is not None and register not in REGISTERS:
            raise ValueError(f"register must be one of {REGISTERS}, got {register!r}")

        def register_of(qubits: tuple[int, ...]) -> str:
            rows = [q <= self.m for q in qubits]
            return "row" if all(rows) else "both" if any(rows) else "column"

        return [
            i
            for i, (step_kind, qubits) in enumerate(self.sorted_steps)
            if kind in (None, step_kind) and register in (None, register_of(qubits))
        ]


def compile_program(gates: list[Gate], m: int, n: int) -> tuple[Program, list[Array]]:
    """The program of a gate sequence on ``m + n`` qubits, and its tensors in stored order."""
    for i, g in enumerate(gates):
        expected = GATE_SHAPES.get(g["kind"])
        if expected is not None and tuple(g["tensor"].shape) != expected:
            raise ValueError(
                f"gate {i} of kind {g['kind']!r} needs a tensor of shape {expected}, "
                f"got {tuple(g['tensor'].shape)}"
            )
    perm = _hadamard_first_perm([g["tensor"] for g in gates])
    slot = [0] * len(gates)
    for position, step in enumerate(perm):
        slot[step] = position
    steps = tuple((g["kind"], tuple(g["qubits"])) for g in gates)
    return Program(m, n, steps, tuple(slot)), [gates[step]["tensor"] for step in perm]


def _axis_of_qubit(q: int, m: int, n: int) -> int:
    """Map a 1-indexed qubit number to its position in the (2,)*N reshape of pic.

    Yao's little-endian convention within each register: qubit `m` maps to axis 0, qubit `m-1` to axis 1,
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
    contraction at all. Measured on an L1 gradient of ``QFTBasis`` on one
    machine: on a GPU slices are 2 to 6 times faster at every size (41 ms
    against 7 ms at 512x512); on a CPU the two are within a factor of two of
    each other, either way depending on the size, and slices compile two to
    three times more slowly. They also change the low bits. Hence opt-in.
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
    arithmetic of the one-qubit gates, see ``_one_qubit``; to use it on a
    basis, construct the basis with ``code=dataclasses.replace(basis.code,
    slices=True)`` and the same for ``inv_code``.
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

    ``apply_circuit`` and ``contract_circuit`` are the single-image entry
    points the bases use; this is the same walk with batch axes and with the
    precision taken from the image.
    """
    m, n = program.m, program.n
    if x.shape[-2:] != (2**m, 2**n):
        raise ValueError(f"image shape must end in ({2**m}, {2**n}), got {x.shape}")
    dtype = jnp.complex64 if x.dtype in (jnp.float32, jnp.complex64) else jnp.complex128
    pic = x.astype(dtype).reshape(x.shape[:-2] + (2,) * (m + n))
    out = _run(program, inverse, slices, tuple(t.astype(dtype) for t in tensors), pic)
    return out.reshape(x.shape)


def register_width(size: int) -> int:
    """The number of qubits of a register that holds ``size`` values; ``size`` must be a power of two."""
    width = size.bit_length() - 1
    if size < 1 or 2**width != size:
        raise ValueError(f"a register holds a power-of-two number of values, got {size}")
    return width


def bit_reverse(x: Array) -> Array:
    """``Pi x Pi``: ``x`` with the bits of its row index and of its column index reversed.

    Yao numbers qubits from the least significant bit and the QFT circuit has
    no final swap layer, so ``QFTBasis`` at its initial tensors is the DFT of
    the bit-reversed image: ``forward_transform(bit_reverse(x))`` equals
    ``ifft2(x, norm="ortho")``. A fixed permutation of the pixels changes
    nothing about how sparse the coefficients are, which is why the bases do
    not undo it. What is defined on the pixels themselves (a sampling mask
    drawn from a seed, a figure) lives in the image's own frame: reverse the
    image going in and the reconstruction coming out to work there.

    An involution. ``x`` has shape ``(..., 2**m, 2**n)``; leading axes are
    batch axes and the register widths are read off the last two.
    """
    m, n = register_width(x.shape[-2]), register_width(x.shape[-1])
    lead = x.ndim - 2
    wires = x.reshape(x.shape[:-2] + (2,) * (m + n))
    # reversing the order of a register's axes reverses the bits of its index
    order = (
        list(range(lead))
        + [lead + axis for axis in reversed(range(m))]
        + [lead + m + axis for axis in reversed(range(n))]
    )
    return jnp.transpose(wires, order).reshape(x.shape)


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


def check_image_shape(pic: Array, m: int, n: int) -> None:
    """Refuse an image that is not ``(2**m, 2**n)``."""
    if pic.shape != (2**m, 2**n):
        raise ValueError(f"pic shape must be (2**m, 2**n) = ({2**m}, {2**n}), got {pic.shape}")


def contract_circuit(tensors: list[Array], code: CircuitCode, m: int, n: int, pic: Array) -> Array:
    """Run ``code`` on ``pic`` laid out one axis per qubit, and give back a ``(2**m, 2**n)`` array.

    Nothing is cast and nothing is checked: the precision of the result is
    what the tensors and the image promote to, and any ``pic`` with
    ``2**(m + n)`` elements is accepted. ``loss_function`` and the Rich,
    RealRich and Blocked transforms have always applied a circuit this way,
    which is why single-precision tensors stay single precision through them.
    """
    out = code(*tensors, pic.reshape((2,) * (m + n)))
    return out.reshape(2**m, 2**n)


def apply_circuit(
    tensors: list[Array],
    code: CircuitCode,
    m: int,
    n: int,
    pic: Array,
) -> Array:
    """Apply ``code`` to one ``(2**m, 2**n)`` image, in double precision.

    ``contract_circuit`` behind a shape check and a cast of the image to
    complex128. Julia's ``ft_mat`` and ``ift_mat``: the inverse is the same
    call with the inverse code and conjugated tensors.
    """
    check_image_shape(pic, m, n)
    return contract_circuit(tensors, code, m, n, pic.astype(jnp.complex128))


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


def extract_phases(tensors: list[Array], gate_indices: list[int]) -> list[float]:
    """The phase of each compact CP tensor at ``gate_indices``, in that order."""
    return [extract_phase_from_cp(tensors[idx]) for idx in gate_indices]


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
