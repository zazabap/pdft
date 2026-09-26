import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.solver as S
from pdft.completion.families.riemannian import dft_matrix, reconstruct_mat
from pdft.completion.transform import synthesis, theta0

n = 5
N = 2**n


def _sparse_problem(seed=0, k=12, p=0.5):
    """A real image exactly 2k-sparse in the DFT domain, half observed."""
    rng = np.random.default_rng(seed)
    C = np.zeros((N, N), complex)
    C.reshape(-1)[rng.choice(N * N, size=k, replace=False)] = rng.standard_normal(
        k
    ) + 1j * rng.standard_normal(k)
    C = C + np.conj(C[(-np.arange(N)) % N][:, (-np.arange(N)) % N])  # Hermitian: real image
    th = theta0(n)
    X = jnp.real(synthesis(jnp.asarray(C), th, th))
    return X, jnp.asarray(rng.random((N, N)) < p), th


def test_kth_largest_static_and_traced_agree():
    mag = jnp.asarray(np.random.default_rng(0).random((8, 8)))
    assert float(S.kth_largest(mag, 5)) == float(jax.jit(S.kth_largest)(mag, 5))
    assert float(S.kth_largest(mag, 5)) == float(np.sort(np.asarray(mag).ravel())[-5])


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
    Xh = S.reconstruct(th, th, Y, obs, 24, 30)
    assert float(jnp.linalg.norm(Xh - X)) < 0.2 * float(jnp.linalg.norm(Y - X))
    assert bool(jnp.all(jnp.where(obs, Xh == Y, True)))


def test_traced_budget_does_not_change_the_result():
    X, obs, th = _sparse_problem()
    a = S.reconstruct(th, th, X * obs, obs, 24, 5)
    b = S.reconstruct(th, th, X * obs, obs, jnp.asarray(24), 5)
    assert jnp.allclose(a, b, atol=1e-12)


def test_batched_matches_per_image():
    X, obs, th = _sparse_problem()
    X2, obs2 = jnp.stack([X, X[::-1]]), jnp.stack([obs, obs[::-1]])
    out = S.reconstruct_batch(th, th, X2 * obs2, obs2, 20, 4)
    for i in range(2):
        assert jnp.allclose(
            out[i], S.reconstruct(th, th, X2[i] * obs2[i], obs2[i], 20, 4), atol=1e-12
        )


def test_the_dense_solver_at_the_dft_matrix_is_the_circuit_solver_at_theta0():
    """solver_for builds every family's solver from its operator alone, so two
    operators that are the same matrix must give the same recovery."""
    X, obs, th = _sparse_problem(seed=5)
    U = dft_matrix(N)
    assert jnp.allclose(
        reconstruct_mat(U, U, X * obs, obs, 20, 6),
        S.reconstruct(th, th, X * obs, obs, 20, 6),
        atol=1e-10,
    )


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_gradient_through_the_solver_is_live_and_finite(mode):
    X, obs, th = _sparse_problem(seed=3)
    th = th + 0.05  # off the exact solution, so the loss has a slope

    def loss(p):
        return jnp.mean((S.reconstruct(p["r"], p["c"], X * obs, obs, 20, 4, mode) - X) ** 2)

    g = jax.grad(loss)({"r": th, "c": th})
    gn = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(g))))
    assert np.isfinite(gn) and gn > 1e-12


def test_rematerialisation_changes_nothing():
    X, obs, th = _sparse_problem(seed=4)

    def loss(p, remat):
        return jnp.mean((S.reconstruct(p, p, X * obs, obs, 20, 4, "hard", remat) - X) ** 2)

    v1, g1 = jax.value_and_grad(loss)(th + 0.05, True)
    v2, g2 = jax.value_and_grad(loss)(th + 0.05, False)
    assert abs(float(v1) - float(v2)) < 1e-12 and float(jnp.abs(g1 - g2).max()) < 1e-9


def test_unknown_mode_raises():
    X, obs, th = _sparse_problem()
    with pytest.raises(KeyError):
        S.reconstruct(th, th, X * obs, obs, 20, 2, "median")
