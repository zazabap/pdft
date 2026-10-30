"""train_basis_exact: a block's environment is the whole transform's, each update is the best, and nothing rises."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.core import program_of, with_tensors
from pdft.loss import topk_truncate
from pdft.training.exact import (
    _PRECISIONS,
    _best,
    _block_matrix,
    _block_objective,
    _block_program,
    _codes,
    block_correlation,
    separable_blocks,
    train_basis_exact,
)
from tests.helpers import BASES, case_rng, complex_normal, generic, random_unitary


def _separable(basis) -> bool:
    try:
        separable_blocks(program_of(basis))
    except ValueError:
        return False
    return True


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
        sub = _block_program(program, qubits, steps)
        u = _block_matrix(sub, [basis.tensors[program.slot[i]] for i in steps])
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


@pytest.mark.parametrize(
    ("template", "real"),
    [
        (pdft.QFTBasis(m=1, n=1).tensors[0], False),  # a Hadamard: U(2)
        (pdft.RichBasis(m=2, n=1).tensors[-1], False),  # a dense two-qubit gate: U(4)
        (pdft.RealRichBasis(m=2, n=1).tensors[-1], True),  # the same, real: O(4)
        (pdft.QFTBasis(m=2, n=1).tensors[-1], False),  # a compact controlled phase: U(1)^4
    ],
    ids=["u2", "u4", "o4", "phase"],
)
def test_the_update_beats_every_other_gate_of_its_manifold(template, real: bool) -> None:
    rng = np.random.default_rng(7)
    environment = complex_normal(rng, template.shape)
    if real:
        environment = environment.real
    phase = template.shape == (2, 2) and bool(np.allclose(np.abs(np.asarray(template)), 1))
    best = np.asarray(_best(template, environment, phase=phase, real=real))

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
    basis = make()
    images = _images(np.random.default_rng(3), basis.m, basis.n, 6)
    budgets = [4, 8]
    seen = []
    result = train_basis_exact(
        basis,
        dataset=images,
        k=budgets,
        sweeps=4,
        passes=2,
        callback=lambda sweep, b, loss: seen.append((sweep, loss)),
    )
    history = result.loss_history + [_objective(result.basis, images, budgets)]
    assert all(b <= a * (1 + 1e-10) for a, b in zip(history, history[1:]))
    assert history[-1] < history[0]
    assert history[0] == pytest.approx(_objective(basis, images, budgets), rel=1e-12)
    assert seen == [(i, loss) for i, loss in enumerate(result.loss_history)]


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


def test_a_sweep_moves_every_block_and_leaves_its_last_gate_at_its_best() -> None:
    """After one sweep the gate it visited last is the best given the rest, in the basis it returns, under its codes.

    That holds only if each block's ``M`` is made after the blocks before it
    were updated, each environment after the gates before it in the block,
    and the last block is trained at all. Every block moving says no block's
    update was dropped. (Not every gate moves: a Hadamard after a dense gate
    on its qubit is at its best once that gate is, which absorbs it.)
    """
    basis = pdft.RichBasis(m=3, n=2)
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
    objective = partial(_block_objective, sub=_block_program(program, qubits, steps))
    environment = np.conj(np.asarray(jax.grad(objective)(block, m)[-1]))
    phase = program.steps[steps[-1]][0] == "CP"
    best = _best(block[-1], environment, phase=phase, real=False)
    np.testing.assert_allclose(best, block[-1], atol=1e-10)


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


def test_the_real_part_is_taken_behind_an_optimization_barrier() -> None:
    """A compile-time property, not a numerical one (CLAUDE.md 14): nothing else would notice it gone."""
    basis = pdft.RichBasis(m=2, n=1)
    program = basis.program
    qubits, steps = separable_blocks(program)[0]
    block = [basis.tensors[program.slot[i]] for i in steps]
    m = jnp.eye(2 ** len(qubits), dtype=jnp.complex128)
    objective = partial(_block_objective, sub=_block_program(program, qubits, steps))
    assert "optimization_barrier" in str(jax.make_jaxpr(jax.grad(objective))(block, m))


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


def test_an_already_sparse_basis_is_a_fixed_point() -> None:
    """Images k-sparse under the basis leave nothing outside the top k, before training and after."""
    basis = pdft.RichBasis(m=2, n=2)
    rng = np.random.default_rng(13)
    shape = (4, 4)
    images = []
    for _ in range(3):
        c = np.zeros(16, complex)
        c[rng.choice(16, 3, replace=False)] = complex_normal(rng, 3)
        images.append(basis.inverse_transform(jnp.asarray(c.reshape(shape))))
    result = train_basis_exact(basis, dataset=images, k=3, sweeps=2)
    assert result.loss_history[0] == pytest.approx(0, abs=1e-20)
    assert _objective(result.basis, images, [3]) == pytest.approx(0, abs=1e-20)


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
        ({"sweeps": 0}, ValueError, "sweeps"),
        ({"passes": 0}, ValueError, "passes"),
        ({"frozen_indices": [99]}, ValueError, "out-of-range"),
        ({"dataset": []}, ValueError, "non-empty"),
        # a Rich transform would reshape a transposed image silently (CLAUDE.md 13)
        ({"dataset": [jnp.ones((2, 4))]}, ValueError, "shape"),
        ({"precision": "half"}, ValueError, "precision"),
    ],
    ids=[
        "k_zero",
        "k_negative",
        "k_empty",
        "k_fractional",
        "no_sweeps",
        "no_passes",
        "frozen_out_of_range",
        "no_images",
        "transposed_image",
        "unknown_precision",
    ],
)
def test_arguments_are_checked(change: dict, error: type, match: str) -> None:
    basis = pdft.RichBasis(m=2, n=1)
    arguments = {"dataset": [jnp.ones((4, 2))], "k": 2, "sweeps": 1} | change
    with pytest.raises(error, match=match):
        train_basis_exact(basis, **arguments)


@pytest.mark.parametrize(("tensors", "atol"), [(jnp.complex128, 1e-12), (jnp.complex64, 1e-6)])
def test_single_precision_follows_double(tensors, atol: float) -> None:
    """The objective in single precision is the one in double to its rounding, over a few sweeps.

    The updates are read from the gates' kinds, not their values, and computed
    in double precision either way: the tensors come back in their own dtype,
    unitary to its rounding. Complex images, so no conjugate pair ties at the
    Fourier start and the two runs make the same choices; complex64 ones, so
    the two runs start from the same numbers and differ only by the arithmetic.
    """
    basis = pdft.RichBasis(m=3, n=2)
    basis = with_tensors(basis, [t.astype(tensors) for t in basis.tensors])
    rng = np.random.default_rng(17)
    images = [x.astype(jnp.complex64) for x in _images(rng, 3, 2, 4, real=False)]
    double = train_basis_exact(basis, dataset=images, k=6, sweeps=3, passes=2)
    single = train_basis_exact(basis, dataset=images, k=6, sweeps=3, passes=2, precision="single")
    np.testing.assert_allclose(single.loss_history, double.loss_history, rtol=1e-5)
    # by rounding, from the first codes on: it did run in single
    assert single.loss_history[0] != double.loss_history[0]
    assert single.loss_history[-1] < single.loss_history[0]
    for t in single.basis.tensors:
        assert t.dtype == tensors
        d = int(np.sqrt(t.size))
        u = np.asarray(t, np.complex128).reshape(d, d)
        assert np.allclose(u @ u.conj().T, np.eye(d), atol=atol)
