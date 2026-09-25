import jax.numpy as jnp
import numpy as np

from pdft.completion.families import riemannian as R
from pdft.completion.transform import analysis, theta0, unitary_matrix

n = 3
N = 2**n


def test_dft_matrix_is_the_circuit_at_theta0():
    assert jnp.allclose(R.dft_matrix(N), unitary_matrix(theta0(n), n), atol=1e-12)


def test_matrix_pair_matches_the_circuit_and_inverts():
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.standard_normal((N, N)))
    U = R.dft_matrix(N)
    C = R.analysis_mat(X, U, U)
    assert jnp.allclose(C, analysis(X, theta0(n), theta0(n), n), atol=1e-12)
    assert jnp.allclose(R.synthesis_mat(C, U, U), X, atol=1e-12)


def test_skew_and_cayley():
    rng = np.random.default_rng(1)
    U = R.dft_matrix(N)
    G = jnp.asarray(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    A = R.skew(U, G)
    assert jnp.allclose(A, -jnp.conj(A).T, atol=1e-12)
    V = R.cayley(U, A, 0.1)
    assert jnp.allclose(jnp.conj(V).T @ V, jnp.eye(N), atol=1e-12)
    assert jnp.allclose(V, U @ (jnp.eye(N) - 0.1 * A), atol=0.1**2 * float(jnp.abs(A).max()) ** 2)


def test_train_unitary_stays_on_the_manifold(capsys):
    rng = np.random.default_rng(2)
    images = rng.random((3, N, N))
    U, hist = R.train_unitary(images, 8, K=2, p=0.5, steps=3, lr=0.05, log_every=1)
    assert len(hist) == 3 and hist[-1]["unitarity"] < 1e-10
    assert not jnp.allclose(U["r"], R.dft_matrix(N))
    assert "mu" in capsys.readouterr().out
    out = R.evaluate_unitary(U, images, 0.5, 0.25, 2, 0)
    assert out.shape == (3,) and np.all(np.isfinite(out))
