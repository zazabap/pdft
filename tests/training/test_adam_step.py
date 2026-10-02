"""The fused Adam step on parameters that are not gate tensors."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pdft.manifolds import EuclideanManifold
from pdft.training.adam_step import adam_stepper, init_adam_moments


def test_the_fused_step_on_flat_parameters_is_plain_adam():
    rng = np.random.default_rng(11)
    start, held = rng.normal(size=5), rng.normal(size=(2, 3))
    scale, shift = rng.normal(size=5), rng.normal(size=5)

    def objective(params, batch):
        p, q = params
        scale, shift = batch
        return jnp.sum(scale * jnp.sin(3.0 * p)) + 0.1 * jnp.sum((p - shift) ** 4) + jnp.sum(q**2)

    lr, beta1, beta2, eps = 5e-3, 0.9, 0.999, 1e-8
    # Adam as Kingma and Ba state it, in numpy, with the gradient written out
    expected, m, v = start.copy(), np.zeros(5), np.zeros(5)
    losses = []
    for t in range(1, 31):
        losses.append(float(objective([expected, held], (scale, shift))))
        g = 3.0 * scale * np.cos(3.0 * expected) + 0.4 * (expected - shift) ** 3
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        expected = expected - lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)

    params = [jnp.asarray(start), jnp.asarray(held)]
    manifolds = [EuclideanManifold(p.shape) for p in params]
    step = adam_stepper(
        objective,
        params,
        manifolds=manifolds,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        max_grad_norm=None,
        frozen_set=frozenset({1}),
    )
    batch = (jnp.asarray(scale), jnp.asarray(shift))
    history = []
    for t in range(1, 31):
        params, loss = step(params, batch, lr, t)
        history.append(float(loss))

    np.testing.assert_allclose(params[0], expected, rtol=1e-13)
    np.testing.assert_allclose(history, losses, rtol=1e-13)  # the loss before each update
    assert params[0].dtype == jnp.float64 and np.abs(expected - start).max() > 0.05
    # the frozen parameter has a gradient and is handed back as it came in
    assert np.asarray(params[1]).tobytes() == held.tobytes()
    # one moment pair per shape: a group is stacked into one array
    m_list, v_list = init_adam_moments(params, manifolds)
    assert [m.shape for m in m_list] == [(5, 1), (2, 3, 1)]
    assert [v.shape for v in v_list] == [(5, 1), (2, 3, 1)]
