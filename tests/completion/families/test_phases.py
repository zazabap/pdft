import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.families import phases as P
from pdft.completion.transform import apply_gates, n_params, theta0, theta_to_params

n = 5
N = 2**n


def test_apply_u_is_the_kernel_at_the_phase_only_parameters(rng, rand_theta):
    th = rand_theta(rng, n)
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    assert jnp.array_equal(P.apply_u(x, th), apply_gates(x, theta_to_params(th)))


def test_theta0_is_the_conjugate_dft_and_mu_is_one():
    U0 = np.asarray(P.unitary_phases(theta0(n)))
    F = np.fft.fft(np.eye(N), axis=0, norm="ortho")
    assert np.abs(U0 - F.conj()).max() < 1e-12
    assert abs(float(P.coherence_phases(theta0(n))) - 1.0) < 1e-12
    assert jax.grad(P.coherence_phases)(theta0(4)).shape == (
        n_params(4),
    )  # traceable, so a loss may carry it


def test_round_trip_and_parseval_at_random_angles(rng, rand_theta):
    thr, thc = rand_theta(rng, n), rand_theta(rng, n)
    X = jnp.asarray(rng.standard_normal((N, N)))
    C = P.analysis(X, thr, thc)
    assert float(jnp.abs(P.synthesis(C, thr, thc) - X).max()) < 1e-11
    assert abs(float(jnp.linalg.norm(C) - jnp.linalg.norm(X))) < 1e-10
    U = np.asarray(P.unitary_phases(thr))
    assert np.abs(U.conj().T @ U - np.eye(N)).max() < 1e-12


def test_reconstruct_recovers_a_sparse_image_and_keeps_observed_pixels(sparse_problem):
    X, obs, th = sparse_problem()
    Y = X * obs
    Xh = P.reconstruct(th, th, Y, obs, 24, 30)
    assert float(jnp.linalg.norm(Xh - X)) < 0.2 * float(jnp.linalg.norm(Y - X))
    assert bool(jnp.all(jnp.where(obs, Xh == Y, True)))


def test_traced_budget_and_batching_change_nothing(sparse_problem):
    X, obs, th = sparse_problem()
    a = P.reconstruct(th, th, X * obs, obs, 24, 5)
    assert jnp.allclose(a, P.reconstruct(th, th, X * obs, obs, jnp.asarray(24), 5), atol=1e-12)
    X2, obs2 = jnp.stack([X, X[::-1]]), jnp.stack([obs, obs[::-1]])
    out = P.reconstruct_batch(th, th, X2 * obs2, obs2, 20, 4)
    for i in range(2):
        assert jnp.allclose(
            out[i], P.reconstruct(th, th, X2[i] * obs2[i], obs2[i], 20, 4), atol=1e-12
        )


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_gradient_through_the_solver_is_live_and_finite(sparse_problem, mode):
    X, obs, th = sparse_problem(seed=3)
    th = th + 0.05  # off the exact solution, so the loss has a slope

    def loss(p):
        return jnp.mean((P.reconstruct(p["r"], p["c"], X * obs, obs, 20, 4, mode) - X) ** 2)

    g = jax.grad(loss)({"r": th, "c": th})
    gn = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(g))))
    assert np.isfinite(gn) and gn > 1e-12


def test_rematerialisation_changes_nothing(sparse_problem):
    X, obs, th = sparse_problem(seed=4)

    def loss(p, remat):
        return jnp.mean((P.reconstruct(p, p, X * obs, obs, 20, 4, "hard", remat) - X) ** 2)

    v1, g1 = jax.value_and_grad(loss)(th + 0.05, True)
    v2, g2 = jax.value_and_grad(loss)(th + 0.05, False)
    assert abs(float(v1) - float(v2)) < 1e-12 and float(jnp.abs(g1 - g2).max()) < 1e-9


def test_unknown_mode_raises(sparse_problem):
    X, obs, th = sparse_problem()
    with pytest.raises(KeyError):
        P.reconstruct(th, th, X * obs, obs, 20, 2, "median")


def test_init_params_and_comp_loss(images):
    p = P.init_params(4)
    assert jnp.array_equal(p["r"], theta0(4)) and jnp.array_equal(p["c"], theta0(4))
    imgs = jnp.asarray(images())
    obs = jnp.asarray(np.random.default_rng(1).random(imgs.shape) < 0.5)
    assert float(P.comp_loss(p, imgs, obs, 20, 2)) == float(
        P.comp_loss(p, imgs, ~obs, 20, 2)
    )  # the mask is ignored


@pytest.mark.parametrize("objective", ["task", "comp"])
def test_train_objectives(images, objective):
    params, hist = P.train(
        images(),
        20,
        K=2,
        p=0.5,
        steps=3,
        lr=1e-2,
        objective=objective,
        lam_mu=0.1,
        log_every=1,
        verbose=False,
    )
    assert set(params) == {"r", "c"} and len(hist) == 3
    assert not jnp.array_equal(params["r"], theta0(4))
    for h in hist:  # mu is pinned whatever the step did
        assert abs(h["mu_r"] - 1.0) < 1e-9 and abs(h["mu_c"] - 1.0) < 1e-9
