"""The parameter views: the controlled-phase angles of a basis as ordinary parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import (
    CP_DIAGONALS,
    CP_PHASES,
    TENSORS,
    bases_allclose,
    cp_diagonals,
    cp_phases,
    program_of,
    with_cp_diagonals,
    with_cp_phases,
)
from pdft.manifolds import EuclideanManifold

from ..helpers import BASES, complex_image, single_precision


def test_cp_phases_reads_the_qft_angles():
    phases = cp_phases(pdft.QFTBasis(m=3, n=2))
    # per register: pi/2 between neighbours, pi/4 one apart
    np.testing.assert_allclose(phases, [np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 2], atol=1e-15)


@pytest.mark.parametrize("case", ["qft_3x2", "entangled_3x2", "tebd_cp_3x2", "mera_cp_4x2"])
def test_with_cp_phases_round_trips(case):
    basis = BASES[case]()
    same = with_cp_phases(basis, cp_phases(basis))
    assert type(same) is type(basis) and pdft.bases_allclose(same, basis, atol=1e-15)
    angles = jnp.asarray(np.random.default_rng(3).uniform(-3, 3, len(cp_phases(basis))))
    moved = with_cp_phases(basis, angles)
    np.testing.assert_allclose(cp_phases(moved), angles, atol=1e-14)
    cp = set(basis.program.tensor_indices(kind="CP"))
    for i, (before, after) in enumerate(zip(basis.tensors, moved.tensors)):
        assert (i in cp) or (after is before)
    x = complex_image(basis.image_size)
    np.testing.assert_allclose(moved.inverse_transform(moved.forward_transform(x)), x, atol=1e-12)


def test_with_cp_phases_is_differentiable_under_jit():
    basis = pdft.QFTBasis(m=2, n=2)
    x = complex_image((4, 4))

    @jax.jit
    def sparsity(angles):
        return jnp.sum(jnp.abs(with_cp_phases(basis, angles).forward_transform(x)))

    angles = cp_phases(basis)
    gradient = jax.grad(sparsity)(angles)
    assert gradient.shape == angles.shape and bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 1e-3
    # and it is the derivative: one finite-difference check per angle
    for k in range(len(angles)):
        step = jnp.zeros_like(angles).at[k].set(1e-6)
        numeric = (sparsity(angles + step) - sparsity(angles - step)) / 2e-6
        assert float(gradient[k]) == pytest.approx(float(numeric), abs=1e-6)


def test_the_view_reaches_through_a_blocked_basis():
    inner = pdft.QFTBasis(m=2, n=2)
    blocked = pdft.BlockedBasis(inner, 1, 1)
    assert program_of(blocked) is inner.program
    np.testing.assert_array_equal(cp_phases(blocked), cp_phases(inner))
    moved = with_cp_phases(blocked, jnp.asarray([0.3, -0.2]))
    assert type(moved) is pdft.BlockedBasis and moved.image_size == (8, 8)
    np.testing.assert_allclose(cp_phases(moved), [0.3, -0.2], atol=1e-15)


def test_bases_without_controlled_phases_have_an_empty_view():
    basis = pdft.RichBasis(m=2, n=2)
    assert cp_phases(basis).shape == (0,) and cp_phases(basis).dtype == jnp.float64
    assert pdft.bases_allclose(with_cp_phases(basis, jnp.zeros((0,))), basis, atol=0.0)
    with pytest.raises(ValueError, match="has 0 controlled-phase gates, got 2 phases"):
        with_cp_phases(basis, jnp.zeros(2))


def test_the_phase_view_is_exact_only_for_tensors_of_the_compact_form():
    """The Riemannian trainers move all four entries of a controlled-phase tensor. The
    view reads one phase and rewrites the tensor in compact form, so on such a basis
    it is a projection, not a round trip."""
    basis = pdft.QFTBasis(m=2, n=2)
    assert bases_allclose(with_cp_phases(basis, cp_phases(basis)), basis, atol=1e-15)

    images = [np.random.default_rng(seed).standard_normal((4, 4)) for seed in range(3)]
    trained = pdft.train_basis_batched(
        basis,
        dataset=images,
        loss=pdft.L1Norm(),
        epochs=3,
        batch_size=3,
        frozen_indices=basis.program.tensor_indices(kind="H"),
    ).basis
    cp = trained.program.tensor_indices(kind="CP")
    assert all(float(jnp.max(jnp.abs(trained.tensors[i][0] - 1.0))) > 1e-4 for i in cp)
    projected = with_cp_phases(trained, cp_phases(trained))
    assert not bases_allclose(projected, trained, atol=1e-6)
    # what it writes it reads back, and writing twice changes nothing more
    np.testing.assert_allclose(cp_phases(projected), cp_phases(trained), atol=1e-15)
    assert bases_allclose(with_cp_phases(projected, cp_phases(projected)), projected, atol=1e-15)
    # the view of all four phases is exact on the trained basis as well
    assert bases_allclose(with_cp_diagonals(trained, cp_diagonals(trained)), trained, atol=1e-15)


def test_cp_phases_are_in_stored_order_entanglers_included():
    phases = [0.3, -1.1]
    basis = pdft.EntangledQFTBasis(m=3, n=2, entangle_phases=phases)
    entanglers = basis.program.tensor_indices(kind="CP", register="both")
    every_cp = basis.program.tensor_indices(kind="CP")
    view = cp_phases(basis)
    assert len(view) == len(every_cp) == 6
    np.testing.assert_allclose([view[every_cp.index(i)] for i in entanglers], phases, atol=1e-15)


def test_cp_diagonals_reads_all_four_phases():
    basis = pdft.QFTBasis(m=3, n=2)
    diagonals = cp_diagonals(basis)
    assert diagonals.shape == (4, 2, 2) and diagonals.dtype == jnp.float64
    # a compact controlled-phase tensor has one phase, in its last entry
    np.testing.assert_array_equal(diagonals[:, 1, 1], cp_phases(basis))
    assert not diagonals.reshape(4, 4)[:, :3].any()
    empty = cp_diagonals(pdft.RichBasis(m=2, n=2))
    assert empty.shape == (0, 2, 2) and empty.dtype == jnp.float64


def test_with_cp_diagonals_writes_every_phase_and_nothing_else():
    basis = pdft.EntangledQFTBasis(m=3, n=2, seed=1)
    angles = jnp.asarray(np.random.default_rng(4).uniform(-3, 3, cp_diagonals(basis).shape))
    moved = with_cp_diagonals(basis, angles)
    np.testing.assert_allclose(cp_diagonals(moved), angles, atol=1e-14)
    cp = basis.program.tensor_indices(kind="CP")
    for i, (before, after) in enumerate(zip(basis.tensors, moved.tensors)):
        if i in cp:
            np.testing.assert_allclose(after, np.exp(1j * angles[cp.index(i)]), atol=1e-15)
        else:
            assert after is before
    x = complex_image(basis.image_size)
    np.testing.assert_allclose(moved.inverse_transform(moved.forward_transform(x)), x, atol=1e-12)
    with pytest.raises(ValueError, match="has 6 controlled-phase gates, got 2 phases"):
        with_cp_diagonals(basis, angles[:2])
    # one angle per gate is the other view's shape
    with pytest.raises(ValueError, match=r"a gate has phases of shape \(2, 2\), got \(\)"):
        with_cp_diagonals(basis, cp_phases(basis))
    with pytest.raises(ValueError, match=r"a gate has phases of shape \(2, 2\), got \(4, 1\)"):
        with_cp_diagonals(basis, angles.reshape(6, 4, 1))


@pytest.mark.parametrize("view", [TENSORS, CP_PHASES, CP_DIAGONALS], ids=lambda v: v.name)
@pytest.mark.parametrize("case", ["qft_3x2", "tebd_cp_3x2", "blocked_qft_2x2_in_3x3"])
def test_a_view_writes_back_what_it_reads(view, case):
    basis = BASES[case]()
    params = view.read(basis)
    assert isinstance(params, list)
    same = view.write(basis, params)
    assert type(same) is type(basis) and bases_allclose(same, basis, atol=1e-15)
    # what the view does not read, it does not replace
    cp = program_of(basis).tensor_indices(kind="CP")
    read = set(range(len(basis.tensors))) if view is TENSORS else set(cp)
    assert all(a is b for i, (a, b) in enumerate(zip(same.tensors, basis.tensors)) if i not in read)


def test_the_views_name_the_geometry_of_their_parameters():
    basis = pdft.QFTBasis(m=3, n=2)
    assert CP_PHASES.manifolds(CP_PHASES.read(basis)) == [EuclideanManifold((4,))]
    assert CP_DIAGONALS.manifolds(CP_DIAGONALS.read(basis)) == [EuclideanManifold((4, 2, 2))]
    # the tensors name none: theirs is read off their values, as it always was
    assert TENSORS.manifolds(TENSORS.read(basis)) is None
    assert repr(CP_PHASES) == "ParameterView(name='cp_phases')"


@pytest.mark.parametrize("view", [CP_PHASES, CP_DIAGONALS], ids=lambda v: v.name)
def test_a_flat_view_reads_in_double_precision_and_writes_in_the_tensors_own(view):
    basis = single_precision(pdft.QFTBasis(m=3, n=2))
    (angles,) = view.read(basis)
    assert angles.dtype == jnp.float64
    written = view.write(basis, [angles + 0.1])
    assert {t.dtype for t in written.tensors} == {jnp.dtype(jnp.complex64)}


@pytest.mark.parametrize("view", [CP_PHASES, CP_DIAGONALS], ids=lambda v: v.name)
def test_a_loss_differentiates_through_a_view(view):
    basis = pdft.QFTBasis(m=2, n=2)
    x = complex_image((4, 4))

    @jax.jit
    def sparsity(params):
        return jnp.sum(jnp.abs(view.write(basis, params).forward_transform(x)))

    (angles,) = view.read(basis)
    (gradient,) = jax.grad(sparsity)([angles])
    assert gradient.shape == angles.shape and gradient.dtype == jnp.float64
    for k in np.ndindex(angles.shape):
        step = jnp.zeros_like(angles).at[k].set(1e-6)
        numeric = (sparsity([angles + step]) - sparsity([angles - step])) / 2e-6
        assert float(gradient[k]) == pytest.approx(float(numeric), abs=1e-6)
    assert float(jnp.max(jnp.abs(gradient))) > 1e-3
