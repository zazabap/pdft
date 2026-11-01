"""Exact block-coordinate training: every gate set to its best given the others, in closed form.

The objective is the sparsity surrogate ``MSELoss`` trains, the energy of the
images outside the ``k`` largest coefficients of their transform,
``sum_i ||x_i - U^H H_k(U x_i)||^2``. For a unitary ``U`` it equals
``sum_i ||U x_i - C_i||^2`` with the codes ``C_i = H_k(U x_i)``, and it is
minimised by alternation, each step exactly, so it never rises (in exact
arithmetic; in floating point, by rounding, once it has converged):

- **Code.** With the circuit fixed, ``C_i = H_k(U x_i)`` (the mean over the
  budgets when there are several).
- **Gates.** With the codes fixed the objective is a constant minus
  ``2 Re sum_i <C_i, U x_i>``, which is linear in the tensor ``g`` of each
  gate: ``Re tr(E^H g)``, ``E`` the gate's environment. The best ``g`` given
  the others is the polar factor of ``E`` for a unitary gate (an orthogonal
  Procrustes problem; in a tensor network, Evenbly and Vidal's update) and
  ``exp(i arg E)``, entry by entry, for a phase tensor (a ``CP`` gate's
  compact form). The gates are visited
  in the order they act, each update seen by the next.

Two matrices learned this way for video coding are Sezer, Cohen and Vetro's
(DCC 2011); the closed-form unitary step and its convergence are Ravishankar
and Bresler's (IEEE TSP 63(9), 2015, arXiv:1501.02859). Here the unitaries are
circuits, each gate on its own manifold, read from its kind in the program.

The circuit must be separable: no gate couples the two registers, so the
transform is a Kronecker product of one unitary per block of qubits that the
gates connect (the two registers of ``QFTBasis``, ``RichBasis`` and
``RealRichBasis``; one block per mode of a multi-mode circuit). A block's part
of the objective is then ``Re tr(M^H U_b)`` with ``M = sum_i C_i Z_i^H`` over
the block's axes, ``Z_i`` the image transformed by every other block (the
basis's own transform, the block's gates set to the identity), and each
environment inside the block is a small computation: the derivative of
``Re tr(M^H U_b)`` by the gate, ``U_b`` the block's gates applied to the
identity by the package's applier.

What is the size of an image (the transforms, the codes, each image's part
of ``M``, a block's unitary and its gradient) is computed in the trainer's
working precision, ``"double"`` or ``"single"``, whatever the basis's own
transforms would cast to (CLAUDE.md 13). ``M`` is summed over the images in
double precision, and each update is computed in double precision from its
environment.
"""

from __future__ import annotations

import dataclasses
import operator
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..bases.core import with_tensors
from ..circuit.builder import (
    CircuitCode,
    Program,
    _axis_of_qubit,
    apply_program,
    contract_circuit,
    identity_tensor,
)
from ..loss import topk_truncate
from .batched import (
    _at_least_one,
    _check_dataset,
    _check_image_shape,
    _validate_frozen_indices,
)
from .result import TrainingResult

Array = jax.Array

# The working precision of everything the size of an image, by name.
_PRECISIONS = {"double": jnp.complex128, "single": jnp.complex64}


def _working_dtype(precision: str):
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {tuple(_PRECISIONS)}, got {precision!r}")
    return _PRECISIONS[precision]


def _count(name: str, value) -> int:
    """``value`` as an integer of at least 1; a bool or a fraction is refused, not rounded."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    _at_least_one(name, value)
    return value


def _budgets(k: int | Sequence[int]) -> list[int]:
    """The budgets ``k`` names, each a positive integer: ``H_0`` keeps nothing, and the updates would wipe the basis."""
    budgets = [k] if np.ndim(k) == 0 else list(k)
    if not budgets:
        raise ValueError("k must name at least one budget")
    return [_count("k", b) for b in budgets]


def _at_most(x: Array, dtype) -> Array:
    """``x`` narrowed to the precision of the complex ``dtype`` if it is wider, real if it is real.

    Widening is left to the arithmetic, which does it exactly at the first
    gate, so an image is never held at more than its own precision.
    """
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact) and jnp.finfo(x.dtype).bits <= jnp.finfo(dtype).bits:
        return x
    return x.astype(dtype if jnp.iscomplexobj(x) else jnp.finfo(dtype).dtype)


def _transform(basis, tensors: Sequence[Array], x: Array, dtype) -> Array:
    """``basis``'s circuit, holding ``tensors``, applied to ``x`` in ``dtype``.

    Through the basis's own code, so with its arithmetic, but in the working
    precision whatever ``forward_transform`` would cast to (CLAUDE.md 13).
    """
    cast = [t.astype(dtype) for t in tensors]
    return contract_circuit(cast, basis.code, basis.m, basis.n, _at_most(x, dtype))


def separable_blocks(program: Program) -> list[tuple[tuple[int, ...], list[int]]]:
    """The blocks of qubits the gates connect, each as ``(qubits, steps)``.

    ``qubits`` are sorted; ``steps`` index ``program.steps`` in the order the
    gates act. A qubit no gate touches forms no block.

    Raises
    ------
    ValueError
        If a gate couples the row and column registers: the circuit is then
        not a product of small unitaries.
    """
    parent = list(range(program.m + program.n + 1))

    def root(q: int) -> int:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for kind, qubits in program.steps:
        rows = [q <= program.m for q in qubits]
        if any(rows) and not all(rows):
            raise ValueError(
                f"a {kind} gate on qubits {qubits} couples the two registers: the circuit is not separable"
            )
        for q in qubits[1:]:
            parent[root(q)] = root(qubits[0])
    blocks: dict[int, list[int]] = {}
    for i, (_, qubits) in enumerate(program.steps):
        blocks.setdefault(root(qubits[0]), []).append(i)
    out = []
    for steps in blocks.values():
        qubits = sorted({q for i in steps for q in program.steps[i][1]})
        out.append((tuple(qubits), steps))
    return sorted(out)


def _block_program(program: Program, qubits: tuple[int, ...], steps: list[int]) -> Program:
    """The block's gates as a one-register circuit on ``len(qubits)`` qubits, its tensors in the order they act.

    The block's qubits keep their order, so the axes of its little circuit run
    as the block's axes do in the whole image (``_axis_of_qubit``).
    """
    local = {q: i + 1 for i, q in enumerate(qubits)}
    sub = tuple((program.steps[i][0], tuple(local[q] for q in program.steps[i][1])) for i in steps)
    return Program(len(qubits), 0, sub, tuple(range(len(steps))))


def _block_matrix(code: CircuitCode, tensors: Sequence[Array], dtype) -> Array:
    """The block's unitary, ``(2**w, 2**w)``, in ``dtype``: column ``j`` its circuit applied to the ``j``-th basis vector.

    ``code`` is the block's own circuit with the basis's arithmetic.
    """
    eye = jnp.eye(2**code.program.m, dtype=dtype)[:, :, None]
    # every option of the code but the circuit and its direction (CLAUDE.md 10)
    arithmetic = {
        f.name: getattr(code, f.name)
        for f in dataclasses.fields(code)
        if f.name not in ("program", "inverse")
    }
    return apply_program(code.program, tensors, eye, **arithmetic)[:, :, 0].T


def _block_objective(tensors: Sequence[Array], m: Array, *, code: CircuitCode) -> Array:
    """``Re tr(M^H U_b)``, a block's part of ``Re sum_i <C_i, U x_i>``; its gradient by a gate is that gate's environment.

    ``U_b`` is computed in the precision of ``m``. The real part is taken
    behind ``optimization_barrier``, as CLAUDE.md 14 asks of any jitted real
    part of a circuit's output: under JAX 0.11.2 XLA carried ``complete``'s
    back through the complex gates, and compiling took exponentially long in
    their depth.
    """
    u = _block_matrix(code, tensors, m.dtype)
    return jnp.real(jax.lax.optimization_barrier(jnp.sum(jnp.conj(m) * u)))


# Compiled once per block circuit, shape and precision, not once per call.
_block_gradient = jax.jit(jax.grad(_block_objective), static_argnames="code")


def _polar(e: np.ndarray, current: np.ndarray, tol: float, scale: float) -> np.ndarray:
    """Of the unitaries ``g`` that maximise ``Re tr(e^H g)``, the one nearest ``current``.

    For ``e = A S B^H`` they are ``A diag(I, W) B^H``, ``W`` any unitary on
    the singular values that are zero (to ``tol`` of the largest, or of
    ``scale`` if that is larger): the polar factor ``A B^H`` when none is,
    ``current`` itself when ``e`` is zero. Otherwise LAPACK would choose ``W``
    by rounding, and the gate would move where the objective cannot see.
    """
    a, s, bh = np.linalg.svd(e)
    rank = int(np.count_nonzero(s > tol * max(s[0], scale)))
    if rank == len(s):
        return a @ bh
    if rank == 0:
        return current
    a0, b0h = a[:, rank:], bh[rank:]
    u, _, vh = np.linalg.svd(a0.conj().T @ current @ b0h.conj().T)
    return a[:, :rank] @ bh[:rank] + a0 @ (u @ vh) @ b0h


def _best(
    tensor: Array,
    environment: np.ndarray,
    *,
    phase: bool,
    real: bool,
    eps: float,
    scale: float = 0.0,
) -> Array:
    """The tensor of a gate's manifold that maximises ``Re tr(E^H g)``, ``E`` its environment.

    ``phase``: a compact controlled phase (kind ``CP``), each entry on the unit
    circle; otherwise a unitary, ``2 x 2`` or a ``(2, 2, 2, 2)`` two-qubit gate.
    The kind comes from the program, not from the values (CLAUDE.md 9), so it
    holds in single precision on a GPU too. Where several tensors are best (an
    entry or a singular value of ``E`` below ``eps`` times the size of ``E``
    times its largest, or times ``scale``, the size of what ``E`` was
    computed from, if that is larger), the one nearest the tensor as it is.
    Computed in double precision and returned in the tensor's.
    """
    e = np.asarray(environment, np.complex128)
    current = np.asarray(tensor, np.complex128)
    if real:
        e, current = np.real(e), np.real(current)
    d = int(np.sqrt(tensor.size))
    if phase:
        magnitude = np.abs(e)
        seen = magnitude > d * eps * max(magnitude.max(), scale)
        best = np.where(seen, e / np.where(seen, magnitude, 1), current)
    else:
        best = _polar(e.reshape(d, d), current.reshape(d, d), d * eps, scale)
        best = best.reshape(tensor.shape)
    return jnp.asarray(best, dtype=tensor.dtype)


def block_correlation(
    basis, steps: Sequence[int], images: Sequence[Array], codes: Sequence[Array]
) -> Array:
    """``M = sum_i C_i Z_i^H`` over a block's axes, ``(2**w, 2**w)``: a block's part of ``Re sum_i <C_i, U x_i>``.

    ``Z_i`` is ``x_i`` under the basis with the block's gates (``steps``,
    indices into ``basis.program.steps``) set to the identity, so that part is
    ``Re tr(M^H U_b)`` with ``U_b`` the block's unitary in the order of
    ``_block_matrix``: rows and columns run over the block's axes as the
    image lays them out, the lowest axis most significant.

    ``Z_i`` and each image's product are computed in the codes' precision;
    their sum, ``M``, is complex128.
    """
    dtype = jnp.result_type(*codes, jnp.complex64)
    program = basis.program
    qubits = sorted({q for i in steps for q in program.steps[i][1]})
    axes = sorted(_axis_of_qubit(q, program.m, program.n) for q in qubits)
    others = [a for a in range(program.m + program.n) if a not in axes]
    identity = list(basis.tensors)
    for i in steps:
        identity[program.slot[i]] = identity_tensor(program.steps[i][0])
    shape = (2,) * (program.m + program.n)
    m = 0.0
    for x, c in zip(images, codes):
        z = _transform(basis, identity, x, dtype)
        # "highest": a single-precision product on a GPU otherwise runs in TF32 (CLAUDE.md 10)
        product = jnp.tensordot(
            c.astype(dtype).reshape(shape),
            jnp.conj(z.reshape(shape)),
            axes=(others, others),
            precision="highest",
        )
        m = m + product.astype(jnp.complex128)
    return m.reshape(2 ** len(axes), 2 ** len(axes))


def _codes(basis, image: Array, budgets: Sequence[int], dtype) -> tuple[Array, float]:
    """``H_k`` of the image's coefficients, the mean over the budgets, and the energy the budgets leave out."""
    c = _transform(basis, basis.tensors, image, dtype)
    codes, tail = 0, 0.0
    for k in budgets:
        kept = topk_truncate(c, k)
        codes = codes + kept
        tail += float(jnp.sum(jnp.abs(c - kept) ** 2, dtype=jnp.float64))
        del kept  # it is the image's size: gone before the next budget's is made
    return codes / len(budgets), tail / len(budgets)


def train_basis_exact(
    basis,
    *,
    dataset: Sequence[Array],
    k: int | Sequence[int],
    sweeps: int,
    passes: int = 1,
    frozen_indices: Sequence[int] | None = None,
    precision: str = "double",
    callback: Callable[[int, object, float], None] | None = None,
) -> TrainingResult:
    """Train a separable circuit basis by exact block-coordinate descent on its sparsity.

    Parameters
    ----------
    dataset
        Images in the basis's frame and of its ``image_size``. They are summed
        as they are; normalise them first for each to weigh the same.
    k
        The coefficient budget, or several, whose objectives are averaged.
    sweeps
        Rounds of the alternation: the codes, then every block's gates.
    passes
        Visits to every gate of a block per sweep, its codes held fixed.
    frozen_indices
        Indices into ``basis.tensors`` that are not updated.
    precision
        ``"double"`` or ``"single"``: the precision of everything the size of
        an image (the images, the transforms, the codes, each image's part of
        ``M``, each block's unitary and its gradient) for every basis and
        whatever the tensors' precision, including the bases whose
        ``forward_transform`` casts to double (CLAUDE.md 13). ``"single"``
        halves the memory these take. ``M`` is summed in double precision and
        each update is computed in double precision from its environment
        either way; the tensors come back in the dtype they came in with,
        unitary to its rounding.
    callback
        ``callback(sweep, basis, loss)`` after every sweep: the basis as the
        sweep left it, and the objective the sweep started from.

    Returns
    -------
    TrainingResult
        Its ``loss_history`` holds, per sweep, the objective at its start:
        the energy outside the top ``k`` summed over the images (the mean over
        the budgets). No sweep raises it in exact arithmetic; once it has
        converged, rounding can, by about ``1e-15`` of it when the working
        precision and the tensors are both double, ``1e-7`` when either is
        single.

    Raises
    ------
    ValueError
        If a gate couples the two registers (``separable_blocks``), the
        dataset or ``k`` is empty, an image is not ``(2**m, 2**n)`` or not
        finite, a budget, ``sweeps`` or ``passes`` is below 1, a frozen index
        is not one of the basis's tensors, or ``precision`` is neither
        ``"double"`` nor ``"single"``.
    TypeError
        For a basis with no circuit of its own (a ``BlockedBasis``), or a
        budget, ``sweeps`` or ``passes`` that is not an integer.

    Notes
    -----
    When every image and every tensor is exactly real, as for a
    ``RealRichBasis`` on real images, each environment is exactly real and
    the updates are computed in real arithmetic: the basis stays exactly
    real. One complex tensor makes every update complex, as under the
    gradient trainers.

    Where the data leave a gate's best value open (several score the same,
    as when the images are zero or their codes already exact), the gate
    takes the one of them nearest its value, and is left as it is when its
    environment is zero; zero here means below the rounding of the working
    precision, next to the block's ``M``.

    The top ``k`` is ``MSELoss``'s, with no tie band (CLAUDE.md 14): where
    coefficients tie in magnitude, as conjugate pairs of a real image do at
    the Fourier start, rounding decides which is kept, and the CPU and the
    GPU, or the two precisions, can follow different trajectories.
    """
    dtype = _working_dtype(precision)
    eps = float(jnp.finfo(dtype).eps)
    program = getattr(basis, "program", None)
    if program is None:
        raise TypeError(
            f"train_basis_exact trains a circuit basis, not a {type(basis).__name__}; "
            "a BlockedBasis tiles one: train its inner basis"
        )
    budgets = _budgets(k)
    sweeps = _count("sweeps", sweeps)
    passes = _count("passes", passes)
    frozen = _validate_frozen_indices(frozen_indices, len(basis.tensors))
    separable = separable_blocks(program)
    _check_dataset(dataset)
    images = []
    for i, x in enumerate(dataset):
        x = _at_most(x, dtype)
        _check_image_shape(i, x, basis.image_size)
        if not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError(f"dataset[{i}] is not finite in {precision} precision")
        images.append(x)
    arithmetic = basis.code if isinstance(basis.code, CircuitCode) else CircuitCode(program)
    blocks = []
    for qubits, steps in separable:
        trainable = [j for j, i in enumerate(steps) if program.slot[i] not in frozen]
        if not trainable:
            continue
        sub = _block_program(program, qubits, steps)
        code = dataclasses.replace(arithmetic, program=sub, inverse=False)
        phase = [program.steps[i][0] == "CP" for i in steps]
        blocks.append((steps, trainable, phase, code))
    real = all(not jnp.iscomplexobj(x) for x in images) and all(
        not np.any(np.imag(np.asarray(t))) for t in basis.tensors
    )

    history: list[float] = []
    start = time.perf_counter()
    for sweep in range(sweeps):
        coded = None  # the last sweep's codes go before this sweep's are made: they are the images' size
        coded = [_codes(basis, x, budgets, dtype) for x in images]
        loss = sum(tail for _, tail in coded)
        history.append(loss)
        for steps, trainable, phase, code in blocks:
            m = block_correlation(basis, steps, images, [codes for codes, _ in coded])
            # every environment of the block is a contraction of M: what is zero is zero next to it
            scale = float(jnp.linalg.norm(m))
            m = m.astype(dtype)  # summed in double; the block's unitary can be as large as an image
            tensors = list(basis.tensors)
            block = [tensors[program.slot[i]] for i in steps]
            for _ in range(passes):
                for j in trainable:
                    # JAX's gradient is the conjugate (CLAUDE.md 1)
                    environment = np.conj(np.asarray(_block_gradient(block, m, code=code)[j]))
                    block[j] = _best(
                        block[j], environment, phase=phase[j], real=real, eps=eps, scale=scale
                    )
            for i, t in zip(steps, block):
                tensors[program.slot[i]] = t
            basis = with_tensors(basis, tensors)
        if callback is not None:
            callback(sweep, basis, loss)
    return TrainingResult(
        basis=basis,
        loss_history=history,
        seed=0,
        steps=sweeps,
        wall_time_s=time.perf_counter() - start,
    )


__all__ = ["block_correlation", "separable_blocks", "train_basis_exact"]
