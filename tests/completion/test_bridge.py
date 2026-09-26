import jax.numpy as jnp
import numpy as np
import pytest

import pdft
import pdft.completion.bridge as Br
from pdft.bases import bases_allclose
from pdft.coherence import coherence as core_mu
from pdft.completion.families.general import (
    analysis_g,
    coherence_general,
    init_general,
    synthesis_g,
    unitary_general,
)
from pdft.completion.families.phases import unitary_phases
from pdft.completion.transform import bitrev_index, theta0
from pdft.io import load_basis, save_basis

m, n = 3, 2


def _image(rng):
    return jnp.asarray(rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n)))


def test_bitrev_image_is_the_two_axis_bit_reversal():
    X = jnp.asarray(np.arange(32).reshape(8, 4))
    out = Br.bitrev_image(X)
    assert np.array_equal(
        np.asarray(out), np.asarray(X)[list(bitrev_index(3))][:, list(bitrev_index(2))]
    )
    assert jnp.array_equal(Br.bitrev_image(out), X)
    with pytest.raises(ValueError):
        Br.bitrev_image(jnp.zeros((6, 4)))


def test_default_basis_is_theta0():
    assert bases_allclose(Br.qft_basis_from_angles(theta0(m), theta0(n)), pdft.QFTBasis(m=m, n=n))


def test_phase_only_identity_and_round_trip(rng, rand_theta):
    thr, thc = rand_theta(rng, m), rand_theta(rng, n)
    basis = Br.qft_basis_from_angles(thr, thc)
    X = _image(rng)
    assert jnp.allclose(
        basis.forward_transform(X),
        unitary_phases(thr).T @ Br.bitrev_image(X) @ unitary_phases(thc),
        atol=1e-12,
    )
    r, c = Br.angles_from_qft_basis(basis)
    assert jnp.allclose(jnp.exp(1j * r), jnp.exp(1j * thr), atol=1e-12)
    assert jnp.allclose(jnp.exp(1j * c), jnp.exp(1j * thc), atol=1e-12)


def test_general_identity_in_both_directions(rng, rand_general):
    pr, pc = rand_general(rng, m), rand_general(rng, n)
    basis = Br.qft_basis_from_general(pr, pc)
    X = _image(rng)
    fwd = basis.forward_transform(X)
    assert jnp.allclose(
        fwd, unitary_general(pr).T @ Br.bitrev_image(X) @ unitary_general(pc), atol=1e-12
    )
    assert jnp.allclose(fwd, jnp.conj(analysis_g(Br.bitrev_image(jnp.conj(X)), pr, pc)), atol=1e-12)
    Xr = jnp.real(X)  # for a real image the conjugation on the input drops out
    assert jnp.allclose(
        basis.forward_transform(Xr), jnp.conj(analysis_g(Br.bitrev_image(Xr), pr, pc)), atol=1e-12
    )
    inv = basis.inverse_transform(fwd)
    assert jnp.allclose(inv, X, atol=1e-11)
    assert jnp.allclose(
        inv, Br.bitrev_image(jnp.conj(synthesis_g(jnp.conj(fwd), pr, pc))), atol=1e-11
    )


def test_general_round_trip_and_coherence(rng, rand_general):
    pr, pc = rand_general(rng, m), rand_general(rng, n)
    basis = Br.qft_basis_from_general(pr, pc)
    qr, qc = Br.general_from_qft_basis(basis)
    for p, q in ((pr, qr), (pc, qc)):
        assert jnp.allclose(p["g"], q["g"], atol=1e-12)
        assert jnp.allclose(jnp.exp(1j * p["phi"]), jnp.exp(1j * q["phi"]), atol=1e-12)
    assert core_mu(basis) == pytest.approx(
        float(coherence_general(pr)) * float(coherence_general(pc)), rel=1e-10
    )
    assert core_mu(Br.qft_basis_from_angles(theta0(m), theta0(n))) == pytest.approx(1.0)


def test_refusals(rng, rand_general):
    general = Br.qft_basis_from_general(rand_general(rng, m), rand_general(rng, n))
    with pytest.raises(ValueError, match="Hadamard"):
        Br.angles_from_qft_basis(general)
    pr = init_general(m)
    pr = {"g": pr["g"], "phi": pr["phi"].at[0, 1].set(0.7)}
    with pytest.raises(ValueError, match="off-"):
        Br.angles_from_qft_basis(Br.qft_basis_from_general(pr, init_general(n)))
    with pytest.raises(TypeError):
        Br.general_from_qft_basis(pdft.TEBDBasis(m=2, n=2))
    broken = Br.qft_basis_from_angles(theta0(m), theta0(n))
    broken.tensors[-1] = broken.tensors[-1] * 2.0
    with pytest.raises(ValueError, match="unit-modulus"):
        Br.general_from_qft_basis(broken)


def test_serialisation_through_the_core_io(tmp_path, rng, rand_theta):
    thr, thc = rand_theta(rng, m), rand_theta(rng, n)
    r, c = Br.angles_from_qft_basis(
        load_basis(save_basis(tmp_path / "basis.json", Br.qft_basis_from_angles(thr, thc)))
    )
    assert jnp.allclose(jnp.exp(1j * r), jnp.exp(1j * thr), atol=1e-10)
    assert jnp.allclose(jnp.exp(1j * c), jnp.exp(1j * thc), atol=1e-10)
