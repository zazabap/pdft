import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.families import general as G
from pdft.completion.transform import apply_u, n_params, theta0, unitary_matrix

n = 5
N = 2**n


def _rand_params(rng, n):
    a = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
    g = jnp.asarray(np.stack([np.linalg.qr(x)[0] for x in a]))
    return {"g": g, "phi": jnp.asarray(rng.uniform(-np.pi, np.pi, (n_params(n), 4)))}


def test_init_general_is_the_phase_only_circuit_at_theta0():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    assert jnp.allclose(
        G.apply_general(x, G.init_general(n), n), apply_u(x, theta0(n), n), atol=1e-12
    )
    assert jnp.allclose(
        G.unitary_general(G.init_general(n), n), unitary_matrix(theta0(n), n), atol=1e-12
    )


def test_adjoint_inverts_at_random_parameters():
    rng = np.random.default_rng(1)
    p = _rand_params(rng, n)
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    y = G.apply_general(x, p, n)
    assert jnp.allclose(G.apply_general(y, p, n, adjoint=True), x, atol=1e-11)
    U = G.unitary_general(p, n)
    assert jnp.allclose(jnp.conj(U).T @ U, jnp.eye(N), atol=1e-11)


def test_rectangular_pair_and_square_solver():
    rng = np.random.default_rng(2)
    nr, nc = 4, 3
    pr, pc = _rand_params(rng, nr), _rand_params(rng, nc)
    X = jnp.asarray(rng.standard_normal((2**nr, 2**nc)))
    C = G.analysis_rect(X, pr, pc, nr, nc)
    assert jnp.allclose(G.synthesis_rect(C, pr, pc, nr, nc), X, atol=1e-11)
    obs = jnp.asarray(rng.random(X.shape) < 0.5)
    out = G.reconstruct_rect(pr, pc, X * obs, obs, nr, nc, 10, 3)
    assert out.shape == X.shape
    p = _rand_params(rng, 4)
    Y = jnp.asarray(rng.standard_normal((16, 16)))
    o = jnp.asarray(rng.random((16, 16)) < 0.5)
    assert jnp.allclose(
        G.reconstruct_g(p, p, Y * o, o, 4, 10, 3), G.reconstruct_rect(p, p, Y * o, o, 4, 4, 10, 3)
    )
    assert jnp.allclose(G.analysis_g(Y, p, p, 4), G.analysis_rect(Y, p, p, 4, 4))
    assert jnp.allclose(G.synthesis_g(Y, p, p, 4), G.synthesis_rect(Y, p, p, 4, 4))


def test_parameter_counts_and_mu():
    assert G.count_params(9) == 4 * 9 + 4 * 36 == 180  # per axis; Table I doubles it
    assert G.count_params_b(9) == 144
    assert abs(float(G.coherence_general(G.init_general(n), n)) - 1.0) < 1e-12
    assert float(G.coherence_general(_rand_params(np.random.default_rng(3), n), n)) > 1.1


def _images(nq=4, count=3, seed=0):
    return np.random.default_rng(seed).random((count, 2**nq, 2**nq)).astype(np.float32)


def test_train_general_models_a_and_b():
    par_b, hist = G.train_general(
        _images(), 4, 16, K=2, p=0.5, steps=3, lr=1e-2, model="B", verbose=False
    )
    assert len(hist) == 3 and set(par_b) == {"r", "c"}
    assert bool(jnp.any(par_b["r"]["phi"][:, :3] != 0))  # B frees all four phases
    par_a, _ = G.train_general(
        _images(), 4, 16, K=2, p=0.5, steps=3, lr=1e-2, model="A", verbose=False
    )
    assert bool(jnp.all(par_a["r"]["phi"][:, :3] == 0))  # A moves only the (1, 1) phase
    assert bool(jnp.any(par_a["r"]["phi"][:, 3] != theta0(4)))
    for par in (par_a, par_b):
        assert jnp.array_equal(par["r"]["g"], G.init_general(4)["g"])  # Hadamards frozen
    with pytest.raises(ValueError):
        G.train_general(_images(), 4, 16, model="C")


def test_train_c_keeps_the_gates_unitary(capsys):
    par = G.train_c(
        _images(), 4, 16, K=2, p=0.5, steps=2, lr_phi=1e-2, lr_g=1e-2, seed=0, log_every=1
    )
    for a in ("r", "c"):
        g = par[a]["g"]
        assert float(jnp.abs(jnp.conj(jnp.swapaxes(g, 1, 2)) @ g - jnp.eye(2)).max()) < 1e-12
        assert not jnp.allclose(g, G.init_general(4)["g"])
    assert "loss" in capsys.readouterr().out


def test_cayley_step_descends_from_a_perturbed_point():
    """A check from exactly theta0 does not expose the sign of the step: the
    first-order term vanishes there and a top-k tie-break jump masks it."""
    nq = 5
    par = {a: G.init_general(nq) for a in ("r", "c")}
    rng = np.random.default_rng(7)
    for a in ("r", "c"):
        S = rng.normal(size=(nq, 2, 2)) + 1j * rng.normal(size=(nq, 2, 2))
        S = 0.5 * (S - np.conj(np.swapaxes(S, 1, 2)))
        Gm = np.asarray(par[a]["g"])
        par[a]["g"] = jnp.asarray(
            np.array([g @ np.asarray(jax.scipy.linalg.expm(0.1 * s)) for g, s in zip(Gm, S)])
        )
        par[a]["phi"] = par[a]["phi"] + 0.1 * rng.normal(size=par[a]["phi"].shape)
    Nq = 2**nq
    obs = jnp.asarray(rng.random((2, Nq, Nq)) < 0.10)
    img = jnp.asarray(np.clip(rng.standard_normal((2, Nq, Nq)) * 0.15 + 0.5, 0, 1))

    def loss(gs):
        pr = {"g": gs["r"], "phi": par["r"]["phi"]}
        pc = {"g": gs["c"], "phi": par["c"]["phi"]}
        Xh = jax.vmap(lambda y, o: G.reconstruct_g(pr, pc, y, o, nq, 64, 10))(img * obs, obs)
        return jnp.mean((Xh - img) ** 2)

    gs = {a: par[a]["g"] for a in ("r", "c")}
    v, gg = jax.value_and_grad(loss)(gs)
    new = {}
    for a in ("r", "c"):
        A = G.riemannian_generator(gs[a], jnp.conj(gg[a]))
        new[a] = jax.vmap(lambda g, aa: G.cayley_u2(g, aa, 1e-3))(gs[a], A)
    assert float(loss(new) - v) < 0
    unit = jnp.abs(jnp.conj(jnp.swapaxes(new["r"], 1, 2)) @ new["r"] - jnp.eye(2)).max()
    assert float(unit) < 1e-12


def test_evaluate_general_runs():
    p = G.init_general(4)
    out = G.evaluate_general({"r": p, "c": p}, _images(), 4, 0.5, 0.25, 3, 0)
    assert out.shape == (3,) and np.all(np.isfinite(out))
