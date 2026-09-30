import jax.numpy as jnp
import numpy as np
import pytest

import pdft.circuit.gates as T
from pdft.coherence import axis_operator

n = 5
N = 2**n


def test_register_bookkeeping():
    assert T.gate_pairs(4) == ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))
    for w in range(1, 8):
        assert len(T.gate_pairs(w)) == T.n_params(w)
        assert T.n_from_params(T.n_params(w)) == w and T.register_width(2**w) == w
    with pytest.raises(ValueError):
        T.n_from_params(4)
    with pytest.raises(ValueError, match="power of two"):
        T.register_width(12)


def test_theta0_is_the_conjugate_dft():
    """The QFT sign convention: the circuit at theta0 is conj(DFT_ortho), not DFT_ortho."""
    p = T.theta_to_params(T.theta0(n))
    U0 = np.asarray(axis_operator(lambda e: T.apply_gates(e, p, axis=0), n))
    F = np.fft.fft(np.eye(N), axis=0, norm="ortho")
    assert np.abs(U0 - F.conj()).max() < 1e-12 and np.abs(U0 - F).max() > 0.1


def test_theta_to_params(rng, rand_theta):
    th = rand_theta(rng, n)
    p = T.theta_to_params(th)
    assert jnp.array_equal(p["g"], T.hadamards(n))
    assert jnp.array_equal(p["phi"][:, 3], th) and not jnp.any(p["phi"][:, :3])


def test_kernel_is_unitary_and_the_adjoint_inverts(rng, rand_general):
    p = rand_general(rng, n)
    U = np.asarray(axis_operator(lambda e: T.apply_gates(e, p, axis=0), n))
    assert np.abs(U.conj().T @ U - np.eye(N)).max() < 1e-11
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    assert jnp.allclose(T.apply_gates(T.apply_gates(x, p), p, adjoint=True), x, atol=1e-11)


def test_four_phase_index_order():
    """Gate ``(p, q)`` reads ``phi[i][2 * b_p + b_q]``. With identity one-qubit
    gates the circuit is that diagonal followed by the bit reversal, so each
    basis vector picks up exactly one slot. The phase-only circuit cannot catch
    a swapped index: it uses slot 3, which is symmetric in the two wires."""
    phi = np.array([[0.1, 0.2, 0.3, 0.4]])
    p = {
        "g": jnp.broadcast_to(jnp.eye(2, dtype=jnp.complex128), (2, 2, 2)),
        "phi": jnp.asarray(phi),
    }
    U = np.asarray(axis_operator(lambda e: T.apply_gates(e, p, axis=0), 2))
    for idx in range(4):
        b_q, b_p = idx >> 1, idx & 1  # the one gate is (p, q) = (1, 0), and wire 0 is the MSB
        assert np.isclose(U[T.bitrev_index(2)[idx], idx], np.exp(1j * phi[0, 2 * b_p + b_q]))
    assert np.count_nonzero(np.abs(U) > 1e-12) == 4


def test_separable_pair_matches_the_dense_matrices_on_a_rectangular_image(rng, rand_general):
    pr, pc = rand_general(rng, 4), rand_general(rng, 3)
    analysis, synthesis = T.separable(T.apply_gates)
    X = jnp.asarray(rng.standard_normal((16, 8)) + 1j * rng.standard_normal((16, 8)))
    Ur = np.asarray(axis_operator(lambda e: T.apply_gates(e, pr, axis=0), 4))
    Uc = np.asarray(axis_operator(lambda e: T.apply_gates(e, pc, axis=0), 3))
    assert np.allclose(synthesis(X, pr, pc), Ur @ np.asarray(X) @ Uc.T, atol=1e-12)
    assert np.allclose(analysis(X, pr, pc), Ur.conj().T @ np.asarray(X) @ Uc.conj(), atol=1e-12)
    assert jnp.allclose(synthesis(analysis(X, pr, pc), pr, pc), X, atol=1e-11)


def test_parameters_for_another_register_are_refused():
    with pytest.raises(ValueError, match="another register"):
        T.apply_gates(jnp.zeros(16), T.theta_to_params(T.theta0(3)))


def test_leading_batch_axes_are_carried_through(rng, rand_general):
    p = rand_general(rng, 4)
    X = jnp.asarray(rng.standard_normal((3, 2, 16)))
    out = T.apply_gates(X, p)
    for i in range(3):
        for j in range(2):
            assert jnp.allclose(out[i, j], T.apply_gates(X[i, j], p), atol=1e-12)


def test_single_precision_input_selects_complex64(rng):
    p = T.theta_to_params(T.theta0(n))
    X = jnp.asarray(rng.random((N, N)), dtype=jnp.float32)
    assert T.complex_dtype(X) == jnp.complex64
    assert T.complex_dtype(jnp.zeros((2,), jnp.float64)) == jnp.complex128
    analysis, synthesis = T.separable(T.apply_gates)
    C = analysis(X, p, p)
    assert C.dtype == jnp.complex64 and float(jnp.abs(synthesis(C, p, p).real - X).max()) < 1e-5


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
    assert T.apply_dense(x, W, axis=0).dtype == jnp.float64  # a real matrix keeps a real image real
    assert T.apply_dense(x.astype(jnp.float32), U, axis=0).dtype == jnp.complex64
    assert T.apply_dense(x.astype(jnp.float32), W, axis=0).dtype == jnp.float32
    # an integer or boolean image must not pull the matrix into its own dtype
    counts = jnp.arange(40).reshape(8, 5)
    assert jnp.allclose(T.apply_dense(counts, W, axis=0), W @ counts.astype(jnp.float64))
    mask = counts % 3 == 0
    assert jnp.allclose(T.apply_dense(mask, W, axis=0), W @ mask.astype(jnp.float64))
