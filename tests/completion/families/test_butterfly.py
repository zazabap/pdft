import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.families import butterfly as B
from pdft.completion.families.phases import apply_u
from pdft.completion.protocol import evaluate_params
from pdft.completion.training import adam_loop, task_loss
from pdft.completion.transform import theta0

n = 4
N = 2**n


@pytest.mark.parametrize("mode", B.MODES)
def test_init_reproduces_the_dft_circuit(rng, mode):
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    p = B.init_butterfly(n, mode)
    assert B.mode_of(p) == mode
    assert jnp.allclose(B.apply_butterfly(x, p), apply_u(x, theta0(n)), atol=1e-12)
    assert abs(float(B.coherence_butterfly(p)) - 1.0) < 1e-12 and B.isometry_defect(p) < 1e-12


def test_expm_and_logm_are_inverse_and_unitary(rng):
    a = jnp.asarray(rng.standard_normal((5, 4)))
    M = B.expm_u2(a)
    assert jnp.allclose(jnp.conj(jnp.swapaxes(M, 1, 2)) @ M, jnp.eye(2), atol=1e-12)
    assert jnp.allclose(B.expm_u2(jnp.asarray(B._logm_u2(np.asarray(M)))), M, atol=1e-12)
    blocks = B.dft_blocks(n)
    assert blocks.shape == (n, N // 2, 2, 2)
    assert np.allclose(np.asarray(B.expm_u2(jnp.asarray(B._logm_u2(blocks)))), blocks, atol=1e-12)
    assert jnp.allclose(B.expm_u2(jnp.zeros((1, 4))), jnp.eye(2))  # the r = 0 branch


@pytest.mark.parametrize("mode", B.MODES)
def test_adjoint_is_the_inverse_after_a_perturbation(rng, mode):
    key = "gen" if mode == "unitary" else "blk"
    p0 = B.init_butterfly(n, mode)
    p = {key: p0[key] + 0.1 * jnp.asarray(rng.standard_normal(p0[key].shape))}
    x = jnp.asarray(rng.standard_normal((N, N)))
    assert jnp.allclose(B.synthesis_b(B.analysis_b(x, p, p), p, p), x, atol=1e-10)
    if mode == "free":
        assert B.isometry_defect(p) > 1e-3 and jnp.allclose(B.blocks(p), p["blk"])


def test_bookkeeping():
    assert B.count_params(9) == 9216 and B.count_params(9, "free") == 18432
    with pytest.raises(KeyError):
        B.mode_of({"foo": 1})
    with pytest.raises(ValueError):
        B.init_butterfly(3, "affine")
    with pytest.raises(ValueError, match="another register"):
        B.apply_butterfly(jnp.zeros(32), B.init_butterfly(n))


def test_free_blocks_descend_through_the_shared_loop(images, rng):
    """The free blocks are the one complex leaf Adam moves. A check from
    exactly the DFT does not expose the direction (top-k ties mask the slope),
    so this starts from a perturbed point and holds the batch fixed."""
    imgs = jnp.asarray(images(n))
    obs = jnp.asarray(rng.random(imgs.shape) < 0.5)
    blk = B.init_butterfly(n, "free")["blk"]
    par = {}
    for a in ("r", "c"):
        noise = rng.standard_normal(blk.shape) + 1j * rng.standard_normal(blk.shape)
        par[a] = {"blk": blk + 0.05 * jnp.asarray(noise)}

    def loss_fn(params, X, o):
        return task_loss(B.reconstruct_butterfly, params, imgs, obs, 16, 4)

    _, hist = adam_loop(imgs, par, loss_fn, lr=1e-3, steps=8, p=0.5, verbose=False)
    assert hist[-1]["loss"] < hist[0]["loss"]


@pytest.mark.parametrize("mode", B.MODES)
def test_solver_training_and_evaluation(images, mode):
    imgs = images(n)
    par, hist = B.train_butterfly(imgs, 16, K=2, p=0.5, steps=3, lr=1e-2, param=mode, verbose=False)
    key = "gen" if mode == "unitary" else "blk"
    assert len(hist) == 3 and "mu_r" in hist[0]
    assert not jnp.allclose(par["r"][key], B.init_butterfly(n, mode)[key])
    X = jnp.asarray(imgs)
    obs = jnp.asarray(np.random.default_rng(3).random(X.shape) < 0.5)
    out = B.reconstruct_butterfly_batch(par["r"], par["c"], X * obs, obs, 16, 2)
    assert jnp.allclose(
        out[0],
        B.reconstruct_butterfly(par["r"], par["c"], X[0] * obs[0], obs[0], 16, 2),
        atol=1e-12,
    )
    ev = evaluate_params(B.reconstruct_butterfly, par, imgs, 0.5, 0.25, 2, 0)
    assert ev.shape == (3,) and np.all(np.isfinite(ev))
