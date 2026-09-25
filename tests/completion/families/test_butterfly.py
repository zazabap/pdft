import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.families import butterfly as B
from pdft.completion.transform import apply_u, theta0

n = 4
N = 2**n


@pytest.mark.parametrize("mode", B.MODES)
def test_init_reproduces_the_dft_circuit(mode):
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    p = B.init_butterfly(n, mode)
    assert B.mode_of(p) == mode
    assert jnp.allclose(B.apply_butterfly(x, p, n), apply_u(x, theta0(n), n), atol=1e-12)
    assert abs(float(B.coherence_butterfly(p, n)) - 1.0) < 1e-12
    assert B.isometry_defect(p, n) < 1e-12


def test_expm_and_logm_are_inverse_and_unitary():
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal((5, 4)))
    M = B.expm_u2(a)
    assert jnp.allclose(jnp.conj(jnp.swapaxes(M, 1, 2)) @ M, jnp.eye(2), atol=1e-12)
    assert jnp.allclose(B.expm_u2(jnp.asarray(B._logm_u2(np.asarray(M)))), M, atol=1e-12)
    blocks = B.dft_blocks(n)
    assert blocks.shape == (n, N // 2, 2, 2)
    assert np.allclose(np.asarray(B.expm_u2(jnp.asarray(B._logm_u2(blocks)))), blocks, atol=1e-12)
    assert jnp.allclose(B.expm_u2(jnp.zeros((1, 4))), jnp.eye(2))  # the r = 0 branch


@pytest.mark.parametrize("mode", B.MODES)
def test_adjoint_is_the_inverse_after_a_perturbation(mode):
    rng = np.random.default_rng(2)
    p = B.init_butterfly(n, mode)
    key = "gen" if mode == "unitary" else "blk"
    p = {key: p[key] + 0.1 * jnp.asarray(rng.standard_normal(p[key].shape))}
    x = jnp.asarray(rng.standard_normal((N, N)))
    C = B.analysis_b(x, p, p, n)
    assert jnp.allclose(B.synthesis_b(C, p, p, n), x, atol=1e-10)
    if mode == "free":
        assert B.isometry_defect(p, n) > 1e-3  # honest: not an isometry any more
        assert jnp.allclose(B.blocks(p), p["blk"])


def test_bookkeeping():
    assert B.count_params(9) == 9216 and B.count_params(9, "free") == 18432
    assert B.n_factors(5) == 5 and B.n_blocks(5) == 16
    with pytest.raises(KeyError):
        B.mode_of({"foo": 1})
    with pytest.raises(ValueError):
        B.init_butterfly(3, "affine")


@pytest.mark.parametrize("mode", B.MODES)
def test_solver_training_and_evaluation(mode):
    rng = np.random.default_rng(3)
    images = rng.random((3, N, N))
    par, hist = B.train_butterfly(
        images, n, 16, K=2, p=0.5, steps=3, lr=1e-2, param=mode, verbose=False
    )
    assert len(hist) == 3 and "mu_r" in hist[0]
    key = "gen" if mode == "unitary" else "blk"
    assert not jnp.allclose(par["r"][key], B.init_butterfly(n, mode)[key])
    X = jnp.asarray(images)
    obs = jnp.asarray(rng.random(X.shape) < 0.5)
    batched = B.reconstruct_butterfly_batch(par["r"], par["c"], X * obs, obs, n, 16, 2)
    single = B.reconstruct_butterfly(par["r"], par["c"], X[0] * obs[0], obs[0], n, 16, 2)
    assert jnp.allclose(batched[0], single, atol=1e-12)
    out = B.evaluate_butterfly(par, images, n, 0.5, 0.25, 2, 0)
    assert out.shape == (3,) and np.all(np.isfinite(out))
