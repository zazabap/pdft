import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.families import general as G
from pdft.completion.families import riemannian as R
from pdft.completion.families.phases import apply_u, unitary_phases
from pdft.completion.protocol import evaluate_params
from pdft.completion.transform import theta0

n = 5
N = 2**n


def test_init_general_is_the_phase_only_circuit_at_theta0(rng):
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    assert jnp.allclose(G.apply_general(x, G.init_general(n)), apply_u(x, theta0(n)), atol=1e-12)
    assert jnp.allclose(G.unitary_general(G.init_general(n)), unitary_phases(theta0(n)), atol=1e-12)


def test_adjoint_inverts_at_random_parameters(rng, rand_general):
    p = rand_general(rng, n)
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    assert jnp.allclose(G.apply_general(G.apply_general(x, p), p, adjoint=True), x, atol=1e-11)
    U = G.unitary_general(p)
    assert jnp.allclose(jnp.conj(U).T @ U, jnp.eye(N), atol=1e-11)


def test_rectangular_pair_and_solver(rng, rand_general):
    pr, pc = rand_general(rng, 4), rand_general(rng, 3)
    X = jnp.asarray(rng.standard_normal((16, 8)))
    assert jnp.allclose(G.synthesis_g(G.analysis_g(X, pr, pc), pr, pc), X, atol=1e-11)
    obs = jnp.asarray(rng.random(X.shape) < 0.5)
    out = G.reconstruct_g(pr, pc, X * obs, obs, 10, 3)
    assert out.shape == X.shape and bool(jnp.all(jnp.where(obs, out == X, True)))


def test_parameter_counts_and_mu(rng, rand_general):
    assert G.count_params(9) == 4 * 9 + 4 * 36 == 180  # per axis; Table I doubles it
    assert G.count_params_b(9) == 144
    assert abs(float(G.coherence_general(G.init_general(n))) - 1.0) < 1e-12
    assert float(G.coherence_general(rand_general(rng, n))) > 1.1


def test_train_general_models_a_and_b(images):
    imgs = images(dtype=np.float32)
    par_b, hist = G.train_general(imgs, 16, K=2, p=0.5, steps=3, lr=1e-2, model="B", verbose=False)
    par_a, _ = G.train_general(imgs, 16, K=2, p=0.5, steps=3, lr=1e-2, model="A", verbose=False)
    assert len(hist) == 3 and set(par_b) == {"r", "c"} and "mu_r" in hist[0]
    assert bool(jnp.any(par_b["r"]["phi"][:, :3] != 0))  # B frees all four phases
    assert bool(jnp.all(par_a["r"]["phi"][:, :3] == 0))  # A moves only the (1, 1) phase
    assert bool(jnp.any(par_a["r"]["phi"][:, 3] != theta0(4)))
    for par in (par_a, par_b):  # the Hadamards are held, bit-exactly
        assert jnp.array_equal(par["r"]["g"], G.init_general(4)["g"])
    with pytest.raises(ValueError):
        G.train_general(imgs, 16, model="C")


def test_train_c_keeps_the_gates_unitary(images, capsys):
    par = G.train_c(
        images(dtype=np.float32),
        16,
        K=2,
        p=0.5,
        steps=2,
        lr_phi=1e-2,
        lr_g=1e-2,
        seed=0,
        log_every=1,
    )
    for a in ("r", "c"):
        g = par[a]["g"]
        assert float(jnp.abs(jnp.conj(jnp.swapaxes(g, 1, 2)) @ g - jnp.eye(2)).max()) < 1e-12
        assert not jnp.allclose(g, G.init_general(4)["g"])
    assert "loss" in capsys.readouterr().out


def test_cayley_step_descends_from_a_perturbed_point():
    """A check from exactly theta0 does not expose the sign of the step: the
    first-order term vanishes there and a top-k tie-break jump masks it."""
    rng = np.random.default_rng(7)
    par = {a: G.init_general(n) for a in ("r", "c")}
    for a in par:
        S = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
        S = 0.5 * (S - np.conj(np.swapaxes(S, 1, 2)))
        par[a]["g"] = par[a]["g"] @ jnp.asarray(
            np.array([np.asarray(jax.scipy.linalg.expm(0.1 * s)) for s in S])
        )
        par[a]["phi"] = par[a]["phi"] + 0.1 * rng.normal(size=par[a]["phi"].shape)
    obs = jnp.asarray(rng.random((2, N, N)) < 0.10)
    img = jnp.asarray(np.clip(rng.standard_normal((2, N, N)) * 0.15 + 0.5, 0, 1))

    def loss(gs):
        p = {a: {"g": gs[a], "phi": par[a]["phi"]} for a in gs}
        Xh = jax.vmap(lambda y, o: G.reconstruct_g(p["r"], p["c"], y, o, 64, 10))(img * obs, obs)
        return jnp.mean((Xh - img) ** 2)

    gs = {a: par[a]["g"] for a in par}
    v, grads = jax.value_and_grad(loss)(gs)
    new = {a: R.cayley(gs[a], R.skew(gs[a], jnp.conj(grads[a])), 1e-3) for a in gs}
    assert float(loss(new) - v) < 0
    assert (
        float(jnp.abs(jnp.conj(jnp.swapaxes(new["r"], 1, 2)) @ new["r"] - jnp.eye(2)).max()) < 1e-12
    )


def test_evaluate_params_runs(images):
    p = G.init_general(4)
    out = evaluate_params(G.reconstruct_g, {"r": p, "c": p}, images(), 0.5, 0.25, 3, 0)
    assert out.shape == (3,) and np.all(np.isfinite(out))
