import jax.numpy as jnp
import numpy as np

from pdft.completion.families import riemannian as R
from pdft.completion.families.phases import analysis, unitary_phases
from pdft.completion.protocol import evaluate_params
from pdft.completion.transform import theta0

n = 3
N = 2**n


def test_dft_matrix_is_the_circuit_at_theta0():
    assert jnp.allclose(R.dft_matrix(N), unitary_phases(theta0(n)), atol=1e-12)


def test_matrix_pair_matches_the_circuit_and_inverts(rng):
    X = jnp.asarray(rng.standard_normal((N, N)))
    U = R.dft_matrix(N)
    C = R.analysis_mat(X, U, U)
    assert jnp.allclose(C, analysis(X, theta0(n), theta0(n)), atol=1e-12)
    assert jnp.allclose(R.synthesis_mat(C, U, U), X, atol=1e-12)


def test_skew_and_cayley_single_and_batched(rng):
    U = R.dft_matrix(N)
    G = jnp.asarray(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    A = R.skew(U, G)
    assert jnp.allclose(A, -jnp.conj(A).T, atol=1e-12)
    V = R.cayley(U, A, 0.1)
    assert jnp.allclose(jnp.conj(V).T @ V, jnp.eye(N), atol=1e-12)
    assert jnp.allclose(V, U @ (jnp.eye(N) - 0.1 * A), atol=0.1**2 * float(jnp.abs(A).max()) ** 2)
    # a batch of U(2) gates: the same functions, no vmap
    g = jnp.asarray(
        np.stack(
            [
                np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))[0]
                for _ in range(4)
            ]
        )
    )
    E = jnp.asarray(rng.standard_normal((4, 2, 2)) + 1j * rng.standard_normal((4, 2, 2)))
    Ab = R.skew(g, E)
    assert Ab.shape == (4, 2, 2) and jnp.allclose(Ab[1], R.skew(g[1], E[1]), atol=1e-12)
    Vb = R.cayley(g, Ab, 0.05)
    assert jnp.allclose(jnp.conj(jnp.swapaxes(Vb, 1, 2)) @ Vb, jnp.eye(2), atol=1e-12)
    assert jnp.allclose(Vb[2], R.cayley(g[2], Ab[2], 0.05), atol=1e-12)


def test_train_unitary_stays_on_the_manifold(images, capsys):
    imgs = images(n)
    U, hist = R.train_unitary(imgs, 8, K=2, p=0.5, steps=3, lr=0.05, log_every=1)
    assert len(hist) == 3 and hist[-1]["unitarity"] < 1e-10 and "mu_r" in hist[-1]
    assert not jnp.allclose(U["r"], R.dft_matrix(N)) and "mu" in capsys.readouterr().out
    out = evaluate_params(R.reconstruct_mat, U, imgs, 0.5, 0.25, 2, 0)  # float32 evaluation
    assert out.shape == (3,) and np.all(np.isfinite(out))
