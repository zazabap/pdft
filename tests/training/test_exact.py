"""train_basis_exact: a block's environment is the whole transform's, each update is the best, and nothing rises."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.core import program_of, with_tensors
from pdft.loss import topk_truncate
from pdft.training.exact import (
    _best,
    _block_matrix,
    _block_program,
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


@pytest.mark.parametrize("case", sorted(BASES))
def test_block_correlation_is_the_transforms_inner_product(case: str) -> None:
    """``Re tr(M^H U_b)`` equals ``Re sum_i <C_i, U x_i>`` for every block, at tensors with no symmetry."""
    basis = BASES[case]()
    if not hasattr(basis, "program") or not _separable(basis):
        pytest.skip("not a separable circuit basis")
    rng = case_rng(case)
    basis = generic(basis, rng)
    program = basis.program
    shape = (2**program.m, 2**program.n)
    images = [jnp.asarray(complex_normal(rng, shape)) for _ in range(2)]
    codes = [jnp.asarray(complex_normal(rng, shape)) for _ in range(2)]
    whole = sum(
        float(jnp.real(jnp.vdot(c, basis.forward_transform(x)))) for x, c in zip(images, codes)
    )
    for qubits, steps in separable_blocks(program):
        m = block_correlation(basis, steps, images, codes)
        sub = _block_program(program, qubits, steps)
        u = _block_matrix(sub, [basis.tensors[program.slot[i]] for i in steps])
        assert float(jnp.real(jnp.sum(jnp.conj(m) * u))) == pytest.approx(whole, rel=1e-10)


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


def test_the_gates_stay_on_their_manifolds() -> None:
    images = _images(np.random.default_rng(5), 3, 2, 4, real=False)
    for basis in (pdft.QFTBasis(m=3, n=2), pdft.RichBasis(m=3, n=2)):
        trained = train_basis_exact(basis, dataset=images, k=6, sweeps=2).basis
        for t in trained.tensors:
            t = np.asarray(t)
            if t.shape == (2, 2) and np.allclose(np.abs(t), 1):
                assert np.allclose(np.abs(t), 1, atol=1e-12)
            else:
                d = int(np.sqrt(t.size))
                u = t.reshape(d, d)
                assert np.allclose(u @ u.conj().T, np.eye(d), atol=1e-12)


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


def test_single_precision_tensors_train_and_stay_unitary() -> None:
    """The update is read from the gate's kind, not its values, and is computed in double precision."""
    basis = pdft.RichBasis(m=3, n=2)
    basis = with_tensors(basis, [t.astype(jnp.complex64) for t in basis.tensors])
    images = [x.astype(jnp.float32) for x in _images(np.random.default_rng(17), 3, 2, 4)]
    result = train_basis_exact(basis, dataset=images, k=6, sweeps=3)
    assert all(t.dtype == jnp.complex64 for t in result.basis.tensors)
    assert result.loss_history[-1] < result.loss_history[0]
    for t in result.basis.tensors:
        d = int(np.sqrt(t.size))
        u = np.asarray(t, np.complex128).reshape(d, d)
        assert np.allclose(u @ u.conj().T, np.eye(d), atol=1e-6)
