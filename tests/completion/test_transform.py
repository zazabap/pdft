import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.transform as T

N_QUBITS = 6
N = 2**N_QUBITS


def test_register_bookkeeping():
    assert T.gate_pairs(4) == ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))
    for n in range(1, 8):
        assert len(T.gate_pairs(n)) == T.n_params(n)
        assert T.n_from_params(T.n_params(n)) == n
        assert T.register_width(2**n) == n
    with pytest.raises(ValueError):
        T.n_from_params(4)
    with pytest.raises(ValueError, match="power of two"):
        T.register_width(12)


def test_theta0_is_the_conjugate_dft():
    """The QFT sign convention: U(theta0) == conj(DFT_ortho), not DFT_ortho."""
    U0 = np.asarray(T.unitary_matrix(T.theta0(N_QUBITS)))
    F = np.fft.fft(np.eye(N), axis=0, norm="ortho")
    assert np.abs(U0 - F.conj()).max() < 1e-12
    assert np.abs(U0 - F).max() > 0.1


def test_theta_to_params_is_the_phase_only_special_case(rng, rand_theta):
    th = rand_theta(rng, 5)
    p = T.theta_to_params(th)
    assert jnp.array_equal(p["g"], T.hadamards(5))
    assert jnp.array_equal(p["phi"][:, 3], th) and not jnp.any(p["phi"][:, :3])
    x = jnp.asarray(rng.standard_normal(32) + 1j * rng.standard_normal(32))
    assert jnp.array_equal(T.apply_u(x, th), T.apply_gates(x, p))


def test_unitary_at_random_angles(rng, rand_theta):
    U = np.asarray(T.unitary_matrix(rand_theta(rng, N_QUBITS)))
    assert np.abs(U.conj().T @ U - np.eye(N)).max() < 1e-12


def test_round_trip_and_parseval(rng, rand_theta):
    thr, thc = rand_theta(rng, N_QUBITS), rand_theta(rng, N_QUBITS)
    X = jnp.asarray(rng.standard_normal((N, N)))
    C = T.analysis(X, thr, thc)
    assert float(jnp.abs(T.synthesis(C, thr, thc) - X).max()) < 1e-11
    assert abs(float(jnp.linalg.norm(C) - jnp.linalg.norm(X))) < 1e-10


def test_matches_the_dense_matrix_on_a_rectangular_image(rng, rand_theta):
    nr, nc = 4, 3
    thr, thc = rand_theta(rng, nr), rand_theta(rng, nc)
    X = jnp.asarray(rng.standard_normal((2**nr, 2**nc)) + 1j * rng.standard_normal((2**nr, 2**nc)))
    Ur, Uc = np.asarray(T.unitary_matrix(thr)), np.asarray(T.unitary_matrix(thc))
    assert np.allclose(T.synthesis(X, thr, thc), Ur @ np.asarray(X) @ Uc.T, atol=1e-12)
    assert np.allclose(T.analysis(X, thr, thc), Ur.conj().T @ np.asarray(X) @ Uc.conj(), atol=1e-12)


def test_parameters_for_another_register_are_refused():
    with pytest.raises(ValueError, match="another register"):
        T.apply_u(jnp.zeros(16), T.theta0(3))


def test_leading_batch_axes_are_carried_through(rng, rand_theta):
    th = rand_theta(rng, 4)
    X = jnp.asarray(rng.standard_normal((3, 2, 16)))
    out = T.apply_u(X, th)
    for i in range(3):
        for j in range(2):
            assert jnp.allclose(out[i, j], T.apply_u(X[i, j], th), atol=1e-12)


def test_single_precision_input_selects_complex64(rng):
    th = T.theta0(5)
    X = jnp.asarray(rng.random((32, 32)), dtype=jnp.float32)
    assert T.complex_dtype(X) == jnp.complex64
    assert T.complex_dtype(jnp.zeros((2,), jnp.float64)) == jnp.complex128
    C = T.analysis(X, th, th)
    assert C.dtype == jnp.complex64
    assert float(jnp.abs(T.synthesis(C, th, th).real - X).max()) < 1e-5


def test_bitreverse():
    assert list(T.bitrev_index(3)) == [0, 4, 2, 6, 1, 5, 3, 7]
    x = jnp.arange(32).reshape(8, 4)
    assert jnp.array_equal(T.bitreverse(x, -2), x[jnp.asarray(T.bitrev_index(3))])
    assert jnp.array_equal(T.bitreverse(T.bitreverse(x)), x)


def test_apply_dense_is_the_matrix_action(rng):
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8)))
    U = jnp.asarray(q)
    x = jnp.asarray(rng.standard_normal((8, 5)))
    assert jnp.allclose(T.apply_dense(x, U, axis=0), U @ x, atol=1e-12)
    assert jnp.allclose(T.apply_dense(x, U, adjoint=True, axis=0), jnp.conj(U).T @ x, atol=1e-12)
    assert jnp.allclose(T.apply_dense(x.T, U, axis=-1), (U @ x).T, atol=1e-12)
    W = jnp.asarray(np.linalg.qr(rng.standard_normal((8, 8)))[0])
    assert T.apply_dense(x, W, axis=0).dtype == jnp.float64  # real stays real
    assert T.apply_dense(x.astype(jnp.float32), U, axis=0).dtype == jnp.complex64


def test_coherence_is_one_at_theta0_and_traceable():
    assert abs(float(T.coherence(T.theta0(5))) - 1.0) < 1e-12
    assert jax.grad(T.coherence)(T.theta0(4)).shape == (T.n_params(4),)
