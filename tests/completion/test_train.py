import jax.numpy as jnp
import numpy as np
import pytest

import pdft.completion.training as TR
from pdft.completion.families.phases import coherence_phases, init_params, reconstruct
from pdft.completion.transform import n_params, theta0

n = 4


def test_widths():
    assert TR.widths(jnp.zeros((3, 16, 8))) == (4, 3)


def test_minibatches_are_the_one_schedule(images):
    imgs = jnp.asarray(images(n, 5))
    a = list(TR.minibatches(imgs, 3, 2, 0.3, seed=7))
    b = list(TR.minibatches(imgs, 3, 2, 0.3, seed=7))
    assert len(a) == 3
    for (Xa, oa), (Xb, ob) in zip(a, b):
        assert Xa.shape == (2, 16, 16) and oa.shape == Xa.shape and oa.dtype == bool
        assert jnp.array_equal(Xa, Xb) and jnp.array_equal(oa, ob)
    assert abs(float(jnp.mean(jnp.stack([o for _, o in a]))) - 0.3) < 0.05
    assert (
        next(TR.minibatches(imgs, 1, 10, 0.3, 0))[0].shape[0] == 5
    )  # the batch is capped at the dataset


def test_adam_loop_trains_and_records(images, capsys):
    def loss_fn(params, X, obs):
        return TR.task_loss(reconstruct, params, X, obs, 20, 3)

    params, hist = TR.adam_loop(
        jnp.asarray(images()),
        init_params(n),
        loss_fn,
        lr=1e-2,
        steps=4,
        p=0.5,
        batch=2,
        monitor=TR.mu_monitor(coherence_phases),
        log_every=2,
    )
    assert len(hist) == 4 and all(np.isfinite(h["loss"]) for h in hist)
    assert "mu_r" in hist[0] and "mu_r" in hist[3] and "mu_r" not in hist[1]
    assert hist[0]["mu_r"] == pytest.approx(1.0) and not jnp.array_equal(params["r"], theta0(n))
    assert "loss" in capsys.readouterr().out


def test_grad_mask_pins_entries(images):
    mask = {"r": jnp.zeros(n_params(n)).at[0].set(1.0), "c": jnp.zeros(n_params(n))}

    def loss_fn(params, X, obs):
        return TR.task_loss(reconstruct, params, X, obs, 20, 2)

    params, _ = TR.adam_loop(
        jnp.asarray(images()),
        init_params(n),
        loss_fn,
        lr=1e-2,
        steps=3,
        p=0.5,
        grad_mask=mask,
        verbose=False,
    )
    assert jnp.array_equal(params["c"], theta0(n)) and jnp.array_equal(
        params["r"][1:], theta0(n)[1:]
    )
    assert params["r"][0] != theta0(n)[0]


def test_non_finite_gradient_raises(images):
    def loss_fn(params, X, obs):
        return jnp.sqrt(jnp.sum(params["r"]) - 1e9)  # nan

    with pytest.raises(FloatingPointError):
        TR.adam_loop(
            jnp.asarray(images()), init_params(n), loss_fn, lr=1e-2, steps=1, p=0.5, verbose=False
        )
