import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.baselines import qtt as Q


def test_quantics_layout_round_trips():
    img = jnp.asarray(np.random.default_rng(0).random((16, 16)))
    t = Q.to_quantics(img)
    assert t.shape == (4, 4, 4, 4)
    assert jnp.array_equal(Q.from_quantics(t), img)


def test_tt_svd_is_exact_on_a_low_rank_train():
    """An outer-product image is not low-rank in the quantics layout (the bits
    of y and x interleave), so the exact case is a train of capped rank."""
    rng = np.random.default_rng(1)
    cores = [jnp.asarray(rng.random(s), dtype=jnp.float32) for s in Q._rank_shapes(4, 4)]
    with jax.default_matmul_precision("highest"):  # no TF32 in the float32 contractions on a GPU
        img = Q.tt_image(cores)
        back = Q.tt_svd(Q.to_quantics(img), 4)
        assert Q.tt_ranks(back) == [1, 4, 4, 4, 1]
        assert Q.tt_nparams(back) == Q.tt_nparams_at(4, 4) == sum(c.size for c in cores)
        assert float(jnp.abs(Q.tt_image(back) - img).max()) < 1e-4 * float(jnp.abs(img).max())
    assert Q.tt_nparams_at(9, 200) == 271136  # the paper's count at 512^2, max_rank 200


def test_masked_pool_and_upsample():
    rng = np.random.default_rng(2)
    img = rng.random((8, 8))
    obs = rng.random((8, 8)) < 0.5
    t1, v1 = Q.masked_pool(img, obs, 1)
    assert np.array_equal(t1, img * obs) and np.array_equal(v1, obs)
    t2, v2 = Q.masked_pool(img, obs, 2)
    assert t2.shape == (4, 4) and v2.dtype == bool
    cell = obs[:2, :2]
    if cell.any():
        assert np.isclose(t2[0, 0], img[:2, :2][cell].mean())
    assert Q.bilinear_up2(img).shape == (16, 16)


def test_level_schedule():
    assert Q.level_schedule(64, 16, 100, (10, 30)) == ([16, 32, 64], [10, 20, 70])
    assert Q.level_schedule(64, 16, 100, (10, 30), coarse_to_fine=False) == ([64], [100])
    assert Q.level_schedule(64, 128, 100, (10,)) == ([64], [100])
    resos, budgets = Q.level_schedule(256, 16, 100, (10,))  # ladder longer than the schedule
    assert resos == [16, 32, 64, 128, 256] and sum(budgets) == 100 and min(budgets) > 0
    with pytest.raises(ValueError):
        Q.level_schedule(64, 16, 20, (10, 30))


@pytest.mark.parametrize("coarse_to_fine", [True, False])
def test_fit_runs_on_a_small_image(coarse_to_fine):
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 16)
    img = np.outer(np.sin(3 * x), np.cos(2 * x)) * 0.5 + 0.5
    obs = rng.random((16, 16)) < 0.5
    cores = Q.fit(
        img,
        obs,
        init_reso=4,
        max_rank=4,
        total_iters=30,
        upsample_at=(8, 16),
        lr=0.05,
        coarse_to_fine=coarse_to_fine,
        verbose=True,
    )
    out = np.asarray(Q.tt_image(cores))
    assert out.shape == (16, 16) and np.all(np.isfinite(out))
    assert np.abs(out[obs] - img[obs]).mean() < np.abs(img[obs]).mean()
