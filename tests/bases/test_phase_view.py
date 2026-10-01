"""`cp_phases` / `with_cp_phases`: the controlled-phase angles of a basis as ordinary parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import bases_allclose, cp_phases, program_of, with_cp_phases
from pdft.bases.circuit.entangled_qft import extract_entangle_phases, get_entangle_tensor_indices

from ..helpers import BASES, complex_image


def test_cp_phases_reads_the_qft_angles():
    phases = cp_phases(pdft.QFTBasis(m=3, n=2))
    # per register: pi/2 between neighbours, pi/4 one apart
    np.testing.assert_allclose(phases, [np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 2], atol=1e-15)


def test_cp_phases_include_the_entanglers_last():
    basis = pdft.EntangledQFTBasis(m=3, n=2, seed=1)
    last = get_entangle_tensor_indices(basis.tensors, basis.n_entangle)
    assert last == basis.program.tensor_indices(kind="CP", register="both")
    np.testing.assert_allclose(
        cp_phases(basis)[-basis.n_entangle :], extract_entangle_phases(basis.tensors, last)
    )


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
    assert cp_phases(basis).shape == (0,)
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
