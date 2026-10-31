"""train_basis_exact: a block's environment is the whole transform's, each update is the best, and nothing rises."""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
import pdft.training.exact as exact
from pdft.bases.core import cp_phases, program_of, with_cp_phases, with_tensors
from pdft.circuit.builder import CircuitCode
from pdft.loss import topk_truncate
from pdft.training.exact import (
    _PRECISIONS,
    _best,
    _block_gradient,
    _block_matrix,
    _block_objective,
    _block_program,
    _codes,
    block_correlation,
    separable_blocks,
    train_basis_exact,
)
from tests.helpers import BASES, case_rng, complex_normal, generic, random_unitary

EPS = float(np.finfo(np.float64).eps)


def _separable(basis) -> bool:
    try:
        separable_blocks(program_of(basis))
    except ValueError:
        return False
    return True


def _block_code(program, qubits, steps) -> CircuitCode:
    return CircuitCode(_block_program(program, qubits, steps))


@pytest.mark.parametrize(("precision", "rel"), [("double", 1e-10), ("single", 1e-5)])
@pytest.mark.parametrize("case", sorted(BASES))
def test_block_correlation_is_the_transforms_inner_product(case: str, precision: str, rel) -> None:
    """``Re tr(M^H U_b)`` equals ``Re sum_i <C_i, U x_i>`` for every block, at tensors with no symmetry.

    In either precision, which the codes are computed in even where
    ``forward_transform`` casts to double (CLAUDE.md 13).
    """
    basis = BASES[case]()
    if not hasattr(basis, "program") or not _separable(basis):
        pytest.skip("not a separable circuit basis")
    rng = case_rng(case)
    basis = generic(basis, rng)
    program = basis.program
    shape = (2**program.m, 2**program.n)
    dtype = _PRECISIONS[precision]
    images = [jnp.asarray(complex_normal(rng, shape)) for _ in range(2)]
    codes = [jnp.asarray(complex_normal(rng, shape), dtype) for _ in range(2)]
    whole = sum(
        float(jnp.real(jnp.vdot(c, basis.forward_transform(x)))) for x, c in zip(images, codes)
    )
    for qubits, steps in separable_blocks(program):
        m = block_correlation(basis, steps, images, codes)
        assert m.dtype == jnp.complex128
        wide = block_correlation(basis, steps, images, [c.astype(jnp.complex128) for c in codes])
        assert bool(jnp.array_equal(m, wide)) == (precision == "double")  # summed in double only
        code = _block_code(program, qubits, steps)
        u = _block_matrix(code, [basis.tensors[program.slot[i]] for i in steps], dtype)
        assert float(jnp.real(jnp.sum(jnp.conj(m) * u))) == pytest.approx(whole, rel=rel)
    assert _codes(basis, images[0], [3], dtype)[0].dtype == dtype


def test_blocks_of_a_qft_are_its_registers() -> None:
    program = pdft.QFTBasis(m=3, n=2).program
    blocks = separable_blocks(program)
    assert [qubits for qubits, _ in blocks] == [(1, 2, 3), (4, 5)]
    assert sorted(i for _, steps in blocks for i in steps) == list(range(len(program.steps)))


@pytest.mark.parametrize("case", ["entangled_3x2", "entangled_front_2x3"])
def test_a_coupled_circuit_is_refused(case: str) -> None:
    """EntangledQFT's entangle layer couples the registers; TEBD and MERA keep to each, and train."""
    basis = BASES[case]()
    image = jnp.ones((2**basis.m, 2**basis.n))
    with pytest.raises(ValueError, match="not separable"):
        train_basis_exact(basis, dataset=[image], k=2, sweeps=1)
    with pytest.raises(ValueError, match="not separable"):  # before an image is read
        train_basis_exact(basis, dataset=[image.T], k=2, sweeps=1)


@pytest.mark.parametrize(
    ("template", "phase", "real"),
    [
        (pdft.QFTBasis(m=1, n=1).tensors[0], False, False),  # a Hadamard: U(2)
        (pdft.RichBasis(m=2, n=1).tensors[-1], False, False),  # a dense two-qubit gate: U(4)
        (pdft.RealRichBasis(m=2, n=1).tensors[-1], False, True),  # the same, real: O(4)
        (pdft.QFTBasis(m=2, n=1).tensors[-1], True, False),  # a compact controlled phase: U(1)^4
    ],
    ids=["u2", "u4", "o4", "phase"],
)
def test_the_update_beats_every_other_gate_of_its_manifold(
    template, phase: bool, real: bool
) -> None:
    rng = np.random.default_rng(7)
    environment = complex_normal(rng, template.shape)
    if real:
        environment = environment.real
    best = np.asarray(_best(template, environment, phase=phase, real=real, eps=EPS))

    def score(g) -> float:
        return float(np.real(np.sum(np.conj(environment) * g)))

    d = int(np.sqrt(template.size))
    for _ in range(200):
        if phase:
            other = np.exp(1j * rng.uniform(0, 2 * np.pi, template.shape))
        else:
            other = random_unitary(rng, d, real=real).reshape(template.shape)
        assert score(best) >= score(other) - 1e-12
    if real:
        assert not np.any(np.imag(best))


@pytest.mark.parametrize("size", [1.0, 1e-20])
def test_of_several_best_gates_the_update_takes_the_nearest(size: float) -> None:
    """Where the environment leaves a choice, the gate moves no further than the best score needs.

    A singular value or an entry of the environment that is zero, or below
    its rounding, says nothing about the data: the polar factor's choice
    there would be LAPACK's, by rounding. Below rounding next to the
    environment's own size, whatever that is, or next to what it was
    computed from: an environment that is all rounding leaves the gate as it
    is.
    """
    rng = np.random.default_rng(47)
    current = random_unitary(rng, 4)
    a, b = random_unitary(rng, 4), random_unitary(rng, 4)
    environment = size * (a @ np.diag([1.0, 0.5, 1e-20, 0.0]) @ b.conj().T)

    def update(e, *, phase=False, scale=0.0, gate=current):
        shape = (2, 2) if phase else (2, 2, 2, 2)
        best = _best(
            jnp.asarray(gate.reshape(shape)),
            e.reshape(shape),
            phase=phase,
            real=False,
            eps=EPS,
            scale=scale,
        )
        return np.asarray(best).reshape(gate.shape)

    best = update(environment)
    np.testing.assert_allclose(best @ best.conj().T, np.eye(4), atol=1e-12)
    score = float(np.real(np.trace(environment.conj().T @ best)))
    assert score == pytest.approx(1.5 * size)
    # every A diag(1, 1, W) B^H scores the same; the update is the one nearest the gate
    for _ in range(100):
        w = np.eye(4, dtype=complex)
        w[2:, 2:] = random_unitary(rng, 2)
        other = a @ w @ b.conj().T
        assert np.linalg.norm(best - current) <= np.linalg.norm(other - current) + 1e-12

    noise = 1e-17 * complex_normal(rng, (4, 4))  # what a block's M of size 1 leaves by rounding
    assert np.array_equal(update(noise, scale=1.0), current)

    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, (2, 2)))
    environment = size * np.array([[2.0, 1e-20j], [0.0, -1.0]])
    best = update(environment, phase=True, gate=phases)
    np.testing.assert_allclose(best[[0, 1], [0, 1]], [1.0, -1.0], atol=1e-15)
    assert best[0, 1] == phases[0, 1] and best[1, 0] == phases[1, 0]
    noise = 1e-17 * complex_normal(rng, (2, 2))
    assert np.array_equal(update(noise, phase=True, scale=1.0, gate=phases), phases)


def _images(rng, m: int, n: int, count: int, *, real: bool = True):
    shape = (2**m, 2**n)
    images = [
        rng.standard_normal(shape) if real else complex_normal(rng, shape) for _ in range(count)
    ]
    return [jnp.asarray(x / np.linalg.norm(x)) for x in images]


def _objective(basis, images, budgets) -> float:
    total = 0.0
    for x in images:
        c = basis.forward_transform(x)
        total += sum(float(jnp.sum(jnp.abs(c - topk_truncate(c, k)) ** 2)) for k in budgets) / len(
            budgets
        )
    return total


@pytest.mark.parametrize(
    "make",
    [
        lambda: pdft.QFTBasis(m=3, n=2),
        lambda: pdft.RichBasis(m=3, n=2),
        lambda: pdft.RealRichBasis(m=2, n=2),
    ],
    ids=["qft", "rich", "real_rich"],
)
def test_the_objective_never_rises(make) -> None:
    """Not by more than rounding, over a few sweeps; the history, the callback and the result agree.

    The images weigh as they are given (one of them twice the others), and
    the callback gets the basis as each sweep left it.
    """
    basis = make()
    images = _images(np.random.default_rng(3), basis.m, basis.n, 6)
    images[0] = 2 * images[0]
    budgets = [4, 8]
    seen = []
    result = train_basis_exact(
        basis,
        dataset=images,
        k=budgets,
        sweeps=4,
        passes=2,
        callback=lambda sweep, b, loss: seen.append((sweep, b, loss)),
    )
    history = result.loss_history + [_objective(result.basis, images, budgets)]
    assert all(b <= a * (1 + 1e-10) for a, b in zip(history, history[1:]))
    assert history[-1] < history[0]
    assert history[0] == pytest.approx(_objective(basis, images, budgets), rel=1e-12)
    assert [(s, loss) for s, _, loss in seen] == list(enumerate(result.loss_history))
    for (_, left, _), after in zip(seen, result.loss_history[1:] + [history[-1]]):
        assert _objective(left, images, budgets) == pytest.approx(after, rel=1e-12)
    assert seen[-1][1] is result.basis
    assert result.steps == 4


@pytest.mark.parametrize("case", sorted(BASES))
def test_each_gate_stays_on_the_manifold_of_its_kind(case: str) -> None:
    """A ``CP`` gate's compact tensor keeps unit-modulus entries, every other gate stays unitary, and so does the transform.

    The kind is read from the program (CLAUDE.md 9): by its values, a ``CP``
    gate updated as a unitary would pass for a one-qubit gate, and its
    transform would shrink the images, which lowers the objective too.
    """
    basis = BASES[case]()
    if not hasattr(basis, "program") or not _separable(basis):
        pytest.skip("not a separable circuit basis")
    rng = case_rng(case)
    images = _images(rng, basis.m, basis.n, 3, real=False)
    size = 2 ** (basis.m + basis.n)
    trained = train_basis_exact(basis, dataset=images, k=size // 4, sweeps=2).basis
    for (kind, _), t in zip(trained.program.sorted_steps, trained.tensors):
        t = np.asarray(t)
        if kind == "CP":
            np.testing.assert_allclose(np.abs(t), 1, atol=1e-12)
        else:
            d = int(np.sqrt(t.size))
            u = t.reshape(d, d)
            np.testing.assert_allclose(u @ u.conj().T, np.eye(d), atol=1e-12)
    x = jnp.asarray(complex_normal(rng, (2**basis.m, 2**basis.n)))
    norm = float(jnp.linalg.norm(trained.forward_transform(x)))
    assert norm == pytest.approx(float(jnp.linalg.norm(x)), rel=1e-12)


def test_a_sweep_visits_the_gates_in_the_order_they_act() -> None:
    """After one sweep, under its codes, the gate it visited last is the best given the rest, and the first no longer is.

    That holds only if the gates are visited in the order they act, each
    block's ``M`` is made after the blocks before it were updated, each
    environment after the gates before it in the block, and the last block is
    trained at all; every block moving says no block's update was dropped. A
    QFT, whose gates cannot absorb their neighbours: in a Rich circuit a
    Hadamard next to a dense gate on its qubit is at its best once that gate
    is, whatever the order.
    """
    basis = pdft.QFTBasis(m=3, n=2)
    images = _images(np.random.default_rng(19), 3, 2, 4, real=False)
    budgets = [4, 8]
    trained = train_basis_exact(basis, dataset=images, k=budgets, sweeps=1).basis
    program = basis.program
    for _, steps in separable_blocks(program):
        slots = [program.slot[i] for i in steps]
        assert not all(np.allclose(basis.tensors[s], trained.tensors[s]) for s in slots)
    codes = [_codes(basis, x, budgets, jnp.complex128)[0] for x in images]
    qubits, steps = separable_blocks(program)[-1]
    m = block_correlation(trained, steps, images, codes)
    block = [trained.tensors[program.slot[i]] for i in steps]
    gradient = _block_gradient(block, m, code=_block_code(program, qubits, steps))

    def at_its_best(j: int) -> bool:
        environment = np.conj(np.asarray(gradient[j]))
        phase = program.steps[steps[j]][0] == "CP"
        best = _best(block[j], environment, phase=phase, real=False, eps=EPS)
        return bool(np.allclose(best, block[j], atol=1e-10))

    assert at_its_best(len(steps) - 1)
    assert not at_its_best(0)


def test_each_pass_descends_on_the_sweeps_codes() -> None:
    """Within a sweep the codes are fixed, and every pass over a block's gates lowers ``sum_i ||U x_i - C_i||^2``.

    One block trains (the column register is frozen), so a run of ``p + 1``
    passes is the run of ``p`` and one more.
    """
    basis = pdft.RichBasis(m=3, n=2)
    frozen = basis.program.tensor_indices(register="column")
    images = _images(np.random.default_rng(23), 3, 2, 4, real=False)
    codes = [_codes(basis, x, [6], jnp.complex128)[0] for x in images]

    def surrogate(b) -> float:
        return sum(
            float(jnp.sum(jnp.abs(b.forward_transform(x) - c) ** 2)) for x, c in zip(images, codes)
        )

    values = [surrogate(basis)] + [
        surrogate(
            train_basis_exact(
                basis, dataset=images, k=6, sweeps=1, passes=p, frozen_indices=frozen
            ).basis
        )
        for p in (1, 2, 3)
    ]
    assert all(b < a - 1e-9 for a, b in zip(values, values[1:])), values


def test_the_codes_are_the_top_k_averaged_over_the_budgets() -> None:
    basis = pdft.QFTBasis(m=2, n=3)
    x = jnp.asarray(complex_normal(np.random.default_rng(29), (4, 8)))
    c = basis.forward_transform(x)
    kept = [topk_truncate(c, k) for k in (3, 10)]
    codes, tail = _codes(basis, x, [3, 10], jnp.complex128)
    np.testing.assert_allclose(codes, (kept[0] + kept[1]) / 2, atol=1e-14)
    left = [float(jnp.sum(jnp.abs(c - h) ** 2)) for h in kept]
    assert tail == pytest.approx(sum(left) / 2, rel=1e-12)


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_a_blocks_objective_is_taken_behind_a_barrier_in_the_precision_of_m(dtype) -> None:
    """Neither shows in a number on a CPU: the barrier is a compile-time property (CLAUDE.md 14), and
    the block's unitary, as large as the image for a two-register circuit with m == n, is memory that
    single precision halves.
    """
    basis = pdft.RichBasis(m=2, n=1)
    program = basis.program
    qubits, steps = separable_blocks(program)[0]
    block = [basis.tensors[program.slot[i]] for i in steps]  # complex128
    m = jnp.eye(2 ** len(qubits), dtype=dtype)
    objective = partial(_block_objective, code=_block_code(program, qubits, steps))
    traced = jax.make_jaxpr(objective)(block, m).jaxpr
    barriers = [e for e in traced.eqns if e.primitive.name == "optimization_barrier"]
    assert [v.aval.dtype for e in barriers for v in e.invars] == [dtype]
    assert "optimization_barrier" in str(jax.make_jaxpr(jax.grad(objective))(block, m))


def test_a_complex_basis_on_real_images_is_trained_as_on_complex_ones() -> None:
    """Only a basis that is exactly real is updated through the real part of its environments.

    At random phases, so that no coefficient of a real image ties with its
    conjugate.
    """
    basis = pdft.QFTBasis(m=3, n=2)
    angles = np.random.default_rng(37).uniform(0, 2 * np.pi, cp_phases(basis).shape)
    basis = with_cp_phases(basis, jnp.asarray(angles))
    images = _images(np.random.default_rng(41), 3, 2, 4)
    real = train_basis_exact(basis, dataset=images, k=6, sweeps=3)
    widened = [x.astype(jnp.complex128) for x in images]
    complex_ = train_basis_exact(basis, dataset=widened, k=6, sweeps=3)
    np.testing.assert_allclose(real.loss_history, complex_.loss_history, rtol=1e-10)
    for a, b in zip(real.basis.tensors, complex_.basis.tensors):
        np.testing.assert_allclose(a, b, atol=1e-10)


def test_a_real_basis_on_real_images_stays_real() -> None:
    basis = pdft.RealRichBasis(m=2, n=3)
    images = _images(np.random.default_rng(9), 2, 3, 4)
    trained = train_basis_exact(basis, dataset=images, k=5, sweeps=3, passes=2).basis
    assert all(not np.any(np.imag(np.asarray(t))) for t in trained.tensors)
    assert _objective(trained, images, [5]) < _objective(basis, images, [5])


def test_frozen_tensors_are_left_as_they_are() -> None:
    basis = pdft.QFTBasis(m=2, n=2)
    frozen = basis.program.tensor_indices(kind="H")
    images = _images(np.random.default_rng(11), 2, 2, 3, real=False)
    trained = train_basis_exact(basis, dataset=images, k=3, sweeps=2, frozen_indices=frozen).basis
    for i, (a, b) in enumerate(zip(basis.tensors, trained.tensors)):
        if i in frozen:
            assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(basis.tensors, trained.tensors)
    )


def _sparse(basis, rng, count: int, nonzero: int):
    """``count`` images with ``nonzero`` coefficients under ``basis``, or zero images for ``nonzero=0``."""
    shape = (2**basis.m, 2**basis.n)
    images = []
    for _ in range(count):
        c = np.zeros(shape[0] * shape[1], complex)
        c[rng.choice(c.size, nonzero, replace=False)] = complex_normal(rng, nonzero)
        images.append(basis.inverse_transform(jnp.asarray(c.reshape(shape))))
    return images


@pytest.mark.parametrize(
    ("make", "count", "nonzero", "k"),
    [
        (lambda: pdft.RichBasis(m=2, n=2), 3, 3, 3),  # environments of rank 1 to 3
        (lambda: pdft.QFTBasis(m=3, n=3), 1, 1, 1),  # zero singular values and phase entries
        (lambda: pdft.QFTBasis(m=2, n=2), 2, 0, 2),  # zero images: every environment is zero
    ],
    ids=["rich_3_sparse", "qft_1_sparse", "zero_images"],
)
@pytest.mark.parametrize(("precision", "atol"), [("double", 1e-12), ("single", 1e-6)])
def test_an_already_sparse_basis_is_a_fixed_point(
    make, count: int, nonzero: int, k: int, precision: str, atol: float
) -> None:
    """Nothing outside the top k, and nothing that would lower it: every gate stays where it is.

    The data do not decide the gates here, and of their best values each gate
    keeps the one it has; a polar factor alone would take one LAPACK chooses
    by rounding, and a zero environment's is the identity. In single
    precision what counts as zero is single precision's rounding.
    """
    basis = make()
    images = _sparse(basis, np.random.default_rng(13), count, nonzero)
    result = train_basis_exact(basis, dataset=images, k=k, sweeps=2, precision=precision)
    assert result.loss_history == pytest.approx([0, 0], abs=1e-10)
    for a, b in zip(basis.tensors, result.basis.tensors):
        np.testing.assert_allclose(b, a, atol=atol)
        if nonzero == 0:  # every value scores the same: left exactly as it is
            assert np.array_equal(np.asarray(a), np.asarray(b))


def test_with_tensors_round_trip_keeps_the_program() -> None:
    """The trained basis is the input basis holding other tensors: same program, same codes."""
    basis = pdft.RichBasis(m=2, n=1)
    trained = train_basis_exact(
        basis, dataset=_images(np.random.default_rng(1), 2, 1, 2), k=2, sweeps=1
    ).basis
    assert trained.program == basis.program
    assert with_tensors(basis, trained.tensors).program == trained.program


def test_a_blocked_basis_is_refused() -> None:
    basis = BASES["blocked_qft_2x2_in_3x3"]()
    with pytest.raises(TypeError, match="inner basis"):
        train_basis_exact(basis, dataset=[jnp.ones((8, 8))], k=2, sweeps=1)


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        # H_0 keeps nothing: every update would then set its gate from a zero environment
        ({"k": 0}, ValueError, "k must be >= 1"),
        ({"k": [3, -1]}, ValueError, "k must be >= 1"),
        ({"k": []}, ValueError, "at least one budget"),
        ({"k": 2.5}, TypeError, "integer"),
        ({"k": True}, TypeError, "integer"),
        ({"sweeps": 0}, ValueError, "sweeps"),
        ({"passes": 0}, ValueError, "passes"),
        ({"passes": 1.5}, TypeError, "passes must be an integer"),
        ({"frozen_indices": [99]}, ValueError, "out-of-range"),
        ({"dataset": []}, ValueError, "non-empty"),
        # a Rich transform would reshape a transposed image silently (CLAUDE.md 13)
        ({"dataset": [jnp.ones((2, 4))]}, ValueError, "shape"),
        # an array is a sequence of its rows
        ({"dataset": jnp.ones((4, 2))}, ValueError, r"dataset\[0\] has shape \(2,\)"),
        ({"dataset": [jnp.ones((4, 2)), jnp.full((4, 2), jnp.nan)]}, ValueError, "finite"),
        ({"precision": "half"}, ValueError, "precision"),
    ],
    ids=[
        "k_zero",
        "k_negative",
        "k_empty",
        "k_fractional",
        "k_bool",
        "no_sweeps",
        "no_passes",
        "fractional_passes",
        "frozen_out_of_range",
        "no_images",
        "transposed_image",
        "one_bare_image",
        "nan_image",
        "unknown_precision",
    ],
)
def test_arguments_are_checked(change: dict, error: type, match: str) -> None:
    basis = pdft.RichBasis(m=2, n=1)
    arguments = {"dataset": [jnp.ones((4, 2))], "k": 2, "sweeps": 1} | change
    with pytest.raises(error, match=match):
        train_basis_exact(basis, **arguments)


@pytest.fixture
def gate_steps(monkeypatch) -> list:
    """Every call the trainer makes to the gate step, as ``(m, code)``."""
    calls = []

    def gradient(block, m, *, code):
        calls.append((m, code))
        return _block_gradient(block, m, code=code)

    monkeypatch.setattr(exact, "_block_gradient", gradient)
    return calls


@pytest.mark.parametrize(("tensors", "atol"), [(jnp.complex128, 1e-12), (jnp.complex64, 1e-6)])
def test_single_precision_follows_double(tensors, atol: float, gate_steps: list) -> None:
    """The objective in single precision is the one in double to its rounding, over a few sweeps.

    The updates are read from the gates' kinds, not their values, and computed
    in double precision either way: the tensors come back in their own dtype,
    unitary to its rounding. Complex images, so no conjugate pair ties at the
    Fourier start and the two runs make the same choices; complex64 ones, so
    the two runs start from the same numbers and differ only by the arithmetic.
    The gate step gets ``M`` in the working precision, whatever the tensors'.
    """
    basis = pdft.RichBasis(m=3, n=2)
    basis = with_tensors(basis, [t.astype(tensors) for t in basis.tensors])
    rng = np.random.default_rng(17)
    images = [x.astype(jnp.complex64) for x in _images(rng, 3, 2, 4, real=False)]
    double = train_basis_exact(basis, dataset=images, k=6, sweeps=3, passes=2)
    assert {m.dtype for m, _ in gate_steps} == {jnp.dtype(jnp.complex128)}
    gate_steps.clear()
    single = train_basis_exact(basis, dataset=images, k=6, sweeps=3, passes=2, precision="single")
    assert {m.dtype for m, _ in gate_steps} == {jnp.dtype(jnp.complex64)}
    np.testing.assert_allclose(single.loss_history, double.loss_history, rtol=1e-5)
    # by rounding, from the first codes on: it did run in single
    assert single.loss_history[0] != double.loss_history[0]
    assert single.loss_history[-1] < single.loss_history[0]
    for t in single.basis.tensors:
        assert t.dtype == tensors
        d = int(np.sqrt(t.size))
        u = np.asarray(t, np.complex128).reshape(d, d)
        assert np.allclose(u @ u.conj().T, np.eye(d), atol=atol)


def test_the_gate_step_keeps_the_bases_arithmetic(gate_steps: list) -> None:
    """A basis built with pdft 0.3.0's arithmetic (CLAUDE.md 10) gets it in its gate step too, not the default."""
    default = pdft.QFTBasis(m=2, n=2)
    contraction = dataclasses.replace(default.code, slices=False, merge_diagonals=False)
    basis = pdft.QFTBasis(m=2, n=2, code=contraction)
    train_basis_exact(basis, dataset=_images(np.random.default_rng(43), 2, 2, 2), k=3, sweeps=1)
    assert gate_steps
    assert all(not (c.slices or c.merge_diagonals or c.inverse) for _, c in gate_steps)
    # and the block's unitary is computed with it: the default slices its Hadamards instead
    qubits, steps = separable_blocks(basis.program)[0]
    block = [basis.tensors[basis.program.slot[i]] for i in steps]

    def contractions(code) -> int:
        traced = jax.make_jaxpr(lambda t: _block_matrix(code, t, jnp.complex128))(block)
        return str(traced).count("dot_general")

    code = gate_steps[0][1]
    assert contractions(code) > 0
    assert contractions(dataclasses.replace(code, slices=True, merge_diagonals=True)) == 0


def test_each_update_is_told_the_size_of_its_blocks_m(gate_steps: list, monkeypatch) -> None:
    """What is zero in an environment is zero next to the ``M`` it was computed from.

    No images found so far leave an environment that is all rounding, so the
    trainer's part is pinned by what it passes.
    """
    scales = []
    best = exact._best

    def recording(*args, scale: float = 0.0, **kwargs):
        scales.append(scale)
        return best(*args, scale=scale, **kwargs)

    monkeypatch.setattr(exact, "_best", recording)
    basis = pdft.QFTBasis(m=2, n=2)
    train_basis_exact(basis, dataset=_images(np.random.default_rng(53), 2, 2, 2), k=3, sweeps=1)
    sizes = [float(jnp.linalg.norm(m)) for m, _ in gate_steps]
    assert scales == pytest.approx(sizes, rel=1e-12)
    assert min(scales) > 0
