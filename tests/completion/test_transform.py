import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.transform as T

N_QUBITS = 6
N = 2**N_QUBITS


def _rand_theta(rng, n):
    return jnp.asarray(rng.uniform(0, 2 * np.pi, size=T.n_params(n)))


def test_gate_pairs_and_counts():
    assert T.gate_pairs(4) == ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))
    for n in range(1, 8):
        assert len(T.gate_pairs(n)) == T.n_params(n)
        assert T.n_from_params(T.n_params(n)) == n
    with pytest.raises(ValueError):
        T.n_from_params(4)


def test_theta0_is_the_conjugate_dft():
    """The QFT sign convention: U(theta0) == conj(DFT_ortho), not DFT_ortho."""
    U0 = np.asarray(T.unitary_matrix(T.theta0(N_QUBITS), N_QUBITS))
    F = np.fft.fft(np.eye(N), axis=0, norm="ortho")
    assert np.abs(U0 - F.conj()).max() < 1e-12
    assert np.abs(U0 - F).max() > 0.1


def test_unitary_at_random_angles():
    rng = np.random.default_rng(0)
    U = np.asarray(T.unitary_matrix(_rand_theta(rng, N_QUBITS), N_QUBITS))
    assert np.abs(U.conj().T @ U - np.eye(N)).max() < 1e-12


def test_round_trip_and_parseval():
    rng = np.random.default_rng(1)
    thr, thc = _rand_theta(rng, N_QUBITS), _rand_theta(rng, N_QUBITS)
    X = jnp.asarray(rng.standard_normal((N, N)))
    C = T.analysis(X, thr, thc, N_QUBITS)
    assert float(jnp.abs(T.synthesis(C, thr, thc, N_QUBITS) - X).max()) < 1e-11
    assert abs(float(jnp.linalg.norm(C) - jnp.linalg.norm(X))) < 1e-10


def test_apply_u_matches_dense_matrix_on_rectangular_image():
    rng = np.random.default_rng(2)
    nr, nc = 4, 3
    thr, thc = _rand_theta(rng, nr), _rand_theta(rng, nc)
    X = jnp.asarray(rng.standard_normal((2**nr, 2**nc)) + 1j * rng.standard_normal((2**nr, 2**nc)))
    Ur = np.asarray(T.unitary_matrix(thr, nr))
    Uc = np.asarray(T.unitary_matrix(thc, nc))
    fwd = T.apply_u(T.apply_u(X, thr, nr, axis=-2), thc, nc, axis=-1)
    assert np.allclose(np.asarray(fwd), Ur @ np.asarray(X) @ Uc.T, atol=1e-12)
    adj = T.apply_u(T.apply_u(X, thr, nr, adjoint=True, axis=-2), thc, nc, adjoint=True, axis=-1)
    assert np.allclose(np.asarray(adj), Ur.conj().T @ np.asarray(X) @ Uc.conj(), atol=1e-12)


def test_leading_batch_axes_are_carried_through():
    rng = np.random.default_rng(3)
    th = _rand_theta(rng, 4)
    X = jnp.asarray(rng.standard_normal((3, 2, 16)))
    out = T.apply_u(X, th, 4, axis=-1)
    for i in range(3):
        for j in range(2):
            assert jnp.allclose(out[i, j], T.apply_u(X[i, j], th, 4), atol=1e-12)


def test_single_precision_input_selects_complex64():
    rng = np.random.default_rng(4)
    th = T.theta0(5)
    X = jnp.asarray(rng.random((32, 32)), dtype=jnp.float32)
    assert T.complex_dtype(X) == jnp.complex64
    assert T.complex_dtype(jnp.zeros((2,), jnp.float64)) == jnp.complex128
    C = T.analysis(X, th, th, 5)
    assert C.dtype == jnp.complex64
    assert float(jnp.abs(T.synthesis(C, th, th, 5).real - X).max()) < 1e-5


def test_coherence_is_one_at_theta0_and_traceable():
    assert abs(float(T.coherence(T.theta0(5), 5)) - 1.0) < 1e-12
    g = jax.grad(lambda th: T.coherence(th, 4))(T.theta0(4))
    assert g.shape == (T.n_params(4),)
