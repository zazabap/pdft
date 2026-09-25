import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.training as TR
from pdft.completion.transform import n_params, theta0

n = 4
N = 2**n


def _images(count=3, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((count, N, N)))


def test_init_params_sits_at_the_dft():
    p = TR.init_params(n)
    assert jnp.array_equal(p["r"], theta0(n)) and jnp.array_equal(p["c"], theta0(n))


def test_adam_loop_trains_and_records(capsys):
    images = _images()

    def loss_fn(params, X, obs):
        return TR.task_loss(params, X, obs, n, 20, 3, "hard")

    params, hist = TR.adam_loop(
        images,
        TR.init_params(n),
        loss_fn,
        lr=1e-2,
        steps=4,
        p=0.5,
        batch=2,
        seed=0,
        monitor=lambda p: {"mu_r": 1.0},
        log_every=2,
        verbose=True,
    )
    assert len(hist) == 4 and all(np.isfinite(h["loss"]) for h in hist)
    assert "mu_r" in hist[0] and "mu_r" in hist[3] and "mu_r" not in hist[1]
    assert not jnp.array_equal(params["r"], theta0(n))
    assert "loss" in capsys.readouterr().out


def test_grad_mask_pins_entries():
    images = _images()
    mask = {"r": jnp.zeros(n_params(n)).at[0].set(1.0), "c": jnp.zeros(n_params(n))}

    def loss_fn(params, X, obs):
        return TR.task_loss(params, X, obs, n, 20, 2, "hard")

    params, _ = TR.adam_loop(
        images, TR.init_params(n), loss_fn, lr=1e-2, steps=3, p=0.5, grad_mask=mask, verbose=False
    )
    assert jnp.array_equal(params["c"], theta0(n))
    assert jnp.array_equal(params["r"][1:], theta0(n)[1:])
    assert params["r"][0] != theta0(n)[0]


def test_non_finite_gradient_raises():
    images = _images()

    def loss_fn(params, X, obs):
        return jnp.sqrt(jnp.sum(params["r"]) - 1e9)  # nan

    with pytest.raises(FloatingPointError):
        TR.adam_loop(images, TR.init_params(n), loss_fn, lr=1e-2, steps=1, p=0.5, verbose=False)


@pytest.mark.parametrize("objective", ["task", "comp"])
def test_train_objectives(objective):
    params, hist = TR.train(
        _images(),
        n,
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
    for h in hist:  # phase-only: mu is pinned whatever the step did
        assert abs(h["mu_r"] - 1.0) < 1e-9 and abs(h["mu_c"] - 1.0) < 1e-9


def test_comp_loss_ignores_the_mask():
    images = _images()
    p = TR.init_params(n)
    obs = jnp.asarray(np.random.default_rng(1).random((3, N, N)) < 0.5)
    a = TR.comp_loss(p, images, obs, n, 20, 2, "hard")
    b = TR.comp_loss(p, images, ~obs, n, 20, 2, "hard")
    assert float(a) == float(b)
