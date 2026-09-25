import jax.numpy as jnp
import numpy as np
import pytest

import pdft
import pdft.completion.bridge as Br
from pdft.bases import bases_allclose
from pdft.coherence import coherence as core_mu
from pdft.completion.families.general import (
    analysis_rect,
    coherence_general,
    init_general,
    synthesis_rect,
    unitary_general,
)
from pdft.completion.transform import n_params, theta0, unitary_matrix
from pdft.io import load_basis, save_basis

m, n = 3, 2


def _rand_general(rng, width):
    a = rng.normal(size=(width, 2, 2)) + 1j * rng.normal(size=(width, 2, 2))
    g = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in a]))
    return {"g": g, "phi": jnp.asarray(rng.uniform(-np.pi, np.pi, (n_params(width), 4)))}


def _bitrev(width):
    return np.array([int(format(i, f"0{width}b")[::-1], 2) for i in range(2**width)])


def test_bitrev_image_is_the_two_axis_bit_reversal():
    X = jnp.asarray(np.arange(32).reshape(8, 4))
    out = Br.bitrev_image(X)
    assert np.array_equal(np.asarray(out), np.asarray(X)[_bitrev(3)][:, _bitrev(2)])
    assert jnp.array_equal(Br.bitrev_image(out), X)
    with pytest.raises(ValueError):
        Br.bitrev_image(jnp.zeros((6, 4)))


def test_default_basis_is_theta0():
    assert bases_allclose(Br.qft_basis_from_angles(theta0(m), theta0(n)), pdft.QFTBasis(m=m, n=n))


def test_phase_only_identity_and_round_trip():
    rng = np.random.default_rng(0)
    thr = jnp.asarray(rng.uniform(0, 2 * np.pi, n_params(m)))
    thc = jnp.asarray(rng.uniform(0, 2 * np.pi, n_params(n)))
    basis = Br.qft_basis_from_angles(thr, thc)
    X = jnp.asarray(rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n)))
    Ur, Uc = unitary_matrix(thr, m), unitary_matrix(thc, n)
    assert jnp.allclose(basis.forward_transform(X), Ur.T @ Br.bitrev_image(X) @ Uc, atol=1e-12)
    r, c = Br.angles_from_qft_basis(basis)
    assert jnp.allclose(jnp.exp(1j * r), jnp.exp(1j * thr), atol=1e-12)
    assert jnp.allclose(jnp.exp(1j * c), jnp.exp(1j * thc), atol=1e-12)


def test_general_identity_in_both_directions():
    rng = np.random.default_rng(1)
    pr, pc = _rand_general(rng, m), _rand_general(rng, n)
    basis = Br.qft_basis_from_general(pr, pc)
    X = jnp.asarray(rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n)))
    fwd = basis.forward_transform(X)
    assert jnp.allclose(
        fwd, jnp.conj(analysis_rect(Br.bitrev_image(jnp.conj(X)), pr, pc, m, n)), atol=1e-12
    )
    Xr = jnp.real(X)  # for a real image the conjugation on the input drops out
    assert jnp.allclose(
        basis.forward_transform(Xr),
        jnp.conj(analysis_rect(Br.bitrev_image(Xr), pr, pc, m, n)),
        atol=1e-12,
    )
    Ur, Uc = unitary_general(pr, m), unitary_general(pc, n)
    assert jnp.allclose(fwd, Ur.T @ Br.bitrev_image(X) @ Uc, atol=1e-12)
    inv = basis.inverse_transform(fwd)
    assert jnp.allclose(inv, X, atol=1e-11)
    # the inverse in the completion subpackage's terms: solve the forward identity
    back = Br.bitrev_image(jnp.conj(synthesis_rect(jnp.conj(fwd), pr, pc, m, n)))
    assert jnp.allclose(inv, back, atol=1e-11)


def test_general_round_trip_and_coherence():
    rng = np.random.default_rng(2)
    pr, pc = _rand_general(rng, m), _rand_general(rng, n)
    basis = Br.qft_basis_from_general(pr, pc)
    qr, qc = Br.general_from_qft_basis(basis)
    for p, q in ((pr, qr), (pc, qc)):
        assert jnp.allclose(p["g"], q["g"], atol=1e-12)
        assert jnp.allclose(jnp.exp(1j * p["phi"]), jnp.exp(1j * q["phi"]), atol=1e-12)
    assert core_mu(basis) == pytest.approx(
        float(coherence_general(pr, m)) * float(coherence_general(pc, n)), rel=1e-10
    )
    assert core_mu(Br.qft_basis_from_angles(theta0(m), theta0(n))) == pytest.approx(1.0)


def test_refusals():
    rng = np.random.default_rng(3)
    general = Br.qft_basis_from_general(_rand_general(rng, m), _rand_general(rng, n))
    with pytest.raises(ValueError, match="Hadamard"):
        Br.angles_from_qft_basis(general)
    pr, pc = init_general(m), init_general(n)
    pr = {"g": pr["g"], "phi": pr["phi"].at[0, 1].set(0.7)}
    with pytest.raises(ValueError, match="off-"):
        Br.angles_from_qft_basis(Br.qft_basis_from_general(pr, pc))
    with pytest.raises(TypeError):
        Br.general_from_qft_basis(pdft.TEBDBasis(m=2, n=2))
    broken = Br.qft_basis_from_angles(theta0(m), theta0(n))
    broken.tensors[-1] = broken.tensors[-1] * 2.0
    with pytest.raises(ValueError, match="unit-modulus"):
        Br.general_from_qft_basis(broken)


def test_serialisation_through_the_core_io(tmp_path):
    rng = np.random.default_rng(4)
    thr = jnp.asarray(rng.uniform(0, 2 * np.pi, n_params(m)))
    thc = jnp.asarray(rng.uniform(0, 2 * np.pi, n_params(n)))
    path = save_basis(tmp_path / "basis.json", Br.qft_basis_from_angles(thr, thc))
    r, c = Br.angles_from_qft_basis(load_basis(path))
    assert jnp.allclose(jnp.exp(1j * r), jnp.exp(1j * thr), atol=1e-10)
    assert jnp.allclose(jnp.exp(1j * c), jnp.exp(1j * thc), atol=1e-10)
