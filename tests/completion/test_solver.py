import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.solver as S
from pdft.completion.transform import synthesis, theta0

n = 5
N = 2**n


def _sparse_problem(seed=0, k=12, p=0.5):
    """An image exactly k-sparse in the DFT domain, half observed."""
    rng = np.random.default_rng(seed)
    C = np.zeros((N, N), complex)
    idx = rng.choice(N * N, size=k, replace=False)
    C.reshape(-1)[idx] = rng.standard_normal(k) + 1j * rng.standard_normal(k)
    C = C + np.conj(C[(-np.arange(N)) % N][:, (-np.arange(N)) % N])  # Hermitian: real image
    th = theta0(n)
    X = jnp.real(synthesis(jnp.asarray(C), th, th, n))
    obs = jnp.asarray(rng.random((N, N)) < p)
    return X, obs, th


def test_kth_largest_static_and_traced_agree():
    mag = jnp.asarray(np.random.default_rng(0).random((8, 8)))
    static = S.kth_largest(mag, 5)
    traced = jax.jit(S.kth_largest)(mag, 5)
    assert float(static) == float(traced) == float(np.sort(np.asarray(mag).ravel())[-5])


def test_hard_k_keeps_exactly_k_entries():
    C = jnp.asarray(np.random.default_rng(1).standard_normal((6, 6)))
    out = S.hard_k(C, 7)
    assert int(jnp.count_nonzero(out)) == 7
    assert jnp.allclose(jnp.where(out != 0, C, 0.0), out)


def test_soft_k_shrinks_towards_zero():
    C = jnp.asarray(np.random.default_rng(2).standard_normal((6, 6)))
    out = S.soft_k(C, 7)
    assert int(jnp.count_nonzero(out)) <= 7
    assert bool(jnp.all(jnp.abs(out) <= jnp.abs(C) + 1e-12))
    assert bool(jnp.all(jnp.sign(out) * jnp.sign(C) >= 0))


def test_iht_recovers_a_sparse_image_and_keeps_observed_pixels():
    X, obs, th = _sparse_problem()
    Y = X * obs
    Xh = S.reconstruct(th, th, Y, obs, n, 24, 30)
    err0 = float(jnp.linalg.norm(Y - X))
    err = float(jnp.linalg.norm(Xh - X))
    assert err < 0.2 * err0
    assert bool(jnp.all(jnp.where(obs, Xh == Y, True)))


def test_traced_budget_does_not_change_the_result():
    X, obs, th = _sparse_problem()
    a = S.reconstruct(th, th, X * obs, obs, n, 24, 5)
    b = S.reconstruct(th, th, X * obs, obs, n, jnp.asarray(24), 5)
    assert jnp.allclose(a, b, atol=1e-12)


def test_reconstruct_batch_matches_per_image():
    X, obs, th = _sparse_problem()
    X2 = jnp.stack([X, X[::-1]])
    obs2 = jnp.stack([obs, obs[::-1]])
    batched = S.reconstruct_batch(th, th, X2 * obs2, obs2, n, 20, 4)
    for i in range(2):
        single = S.reconstruct(th, th, X2[i] * obs2[i], obs2[i], n, 20, 4)
        assert jnp.allclose(batched[i], single, atol=1e-12)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_gradient_through_the_solver_is_live_and_finite(mode):
    X, obs, th = _sparse_problem(seed=3)
    th = th + 0.05  # off the exact solution, so the loss has a slope

    def loss(p):
        Xh = S.reconstruct(p["r"], p["c"], X * obs, obs, n, 20, 4, mode)
        return jnp.mean((Xh - X) ** 2)

    g = jax.grad(loss)({"r": th, "c": th})
    gn = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(g))))
    assert np.isfinite(gn) and gn > 1e-12


def test_rematerialisation_changes_nothing():
    X, obs, th = _sparse_problem(seed=4)

    def loss(p, remat):
        Xh = S.reconstruct(p, p, X * obs, obs, n, 20, 4, "hard", remat)
        return jnp.mean((Xh - X) ** 2)

    v1, g1 = jax.value_and_grad(loss)(th + 0.05, True)
    v2, g2 = jax.value_and_grad(loss)(th + 0.05, False)
    assert abs(float(v1) - float(v2)) < 1e-12
    assert float(jnp.abs(g1 - g2).max()) < 1e-9


def test_unknown_mode_raises():
    X, obs, th = _sparse_problem()
    with pytest.raises(KeyError):
        S.reconstruct(th, th, X * obs, obs, n, 20, 2, "median")


def test_evaluate_theta_returns_one_psnr_per_image():
    X, obs, th = _sparse_problem()
    out = S.evaluate_theta({"r": th, "c": th}, [np.asarray(X), np.asarray(X.T)], n, 0.5, 0.05, 6, 0)
    assert out.shape == (2,)
    assert np.all(out > 10)
