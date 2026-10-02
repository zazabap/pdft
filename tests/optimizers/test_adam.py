"""RiemannianAdam unit + integration tests."""

import jax
import jax.numpy as jnp
import numpy as np

import pdft
from pdft.optimizers import RiemannianAdam, optimize

from ..helpers import PlainAdam


def test_riemannian_adam_defaults_match_upstream():
    opt = RiemannianAdam()
    assert opt.lr == 0.001
    assert opt.beta1 == 0.9
    assert opt.beta2 == 0.999
    assert opt.eps == 1e-8
    assert opt.max_grad_norm is None


def _random_unitary(d: int, count: int, key=jax.random.PRNGKey(0)) -> list:
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (count, d, d)) + 1j * jax.random.normal(k2, (count, d, d))
    Q, _ = jnp.linalg.qr(A)
    return [Q[i].astype(jnp.complex128) for i in range(count)]


def test_adam_preserves_unitarity_across_iters():
    tensors = _random_unitary(d=4, count=3)

    def loss_fn(ts):
        return sum(jnp.real(jnp.trace(t)) for t in ts)

    grad_fn = jax.grad(loss_fn)
    opt = RiemannianAdam(lr=0.05)
    final, trace = optimize(
        opt, tensors, loss_fn, grad_fn, max_iter=50, tol=1e-10, record_loss=True
    )

    for t in final:
        d = t.shape[0]
        assert jnp.allclose(t @ jnp.conj(t).T, jnp.eye(d), atol=1e-6)

    # Adam is not monotone but should make net progress
    losses = jnp.array(trace)
    assert losses[-1] < losses[0]


def test_adam_reduces_loss_on_training():
    import numpy as np

    from pdft.bases.base import QFTBasis
    from pdft.loss import L1Norm
    from pdft.training import train_basis

    target = jax.random.normal(jax.random.PRNGKey(3), (4, 4)).astype(jnp.complex128)
    basis = QFTBasis(m=2, n=2)
    result = train_basis(
        basis,
        target=target,
        loss=L1Norm(),
        optimizer=RiemannianAdam(lr=0.01),
        steps=30,
        seed=0,
    )
    losses = np.asarray(result.loss_history)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


def test_adam_update_is_the_textbook_step_on_the_manifold():
    from pdft.manifolds import PhaseManifold
    from pdft.optimizers.adam import _adam_update, _zero_moments
    from pdft.training.adam_step import init_adam_moments

    rng = np.random.default_rng(0)
    points = jnp.asarray(np.exp(1j * rng.uniform(-3, 3, (2, 2, 3))))
    manifold = PhaseManifold()
    rgrad = manifold.project(points, jnp.asarray(rng.normal(size=(2, 2, 3)) + 0j))
    m0, v0 = _zero_moments(points)
    assert m0.dtype == points.dtype and v0.dtype == jnp.float64 and not m0.any() and not v0.any()
    new_points, m1, v1 = _adam_update(
        manifold,
        points,
        rgrad,
        m0,
        v0,
        lr=0.1,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bc1=0.1,
        bc2=0.001,
    )
    np.testing.assert_allclose(v1, 0.001 * np.abs(rgrad) ** 2, rtol=1e-14)
    direction = rgrad / (
        np.abs(rgrad) + 1e-8
    )  # (m / bc1) / (sqrt(v / bc2) + eps) at the first step
    np.testing.assert_allclose(new_points, manifold.retract(points, -direction, 0.1), atol=1e-12)
    np.testing.assert_allclose(m1, manifold.transport(points, new_points, 0.1 * rgrad), atol=1e-14)

    # one moment pair per manifold group, shaped like the stacked group
    basis = pdft.RichBasis(m=2, n=2)
    m_list, v_list = init_adam_moments(basis.tensors)
    assert [m.shape for m in m_list] == [(2, 2, 4), (2, 2, 2, 2, 2)]
    assert [v.shape for v in v_list] == [(2, 2, 4), (2, 2, 2, 2, 2)]


def test_adam_update_on_flat_parameters_is_plain_adam():
    from pdft.manifolds import EuclideanManifold
    from pdft.optimizers.adam import _adam_update, _zero_moments

    weights = np.arange(1.0, 8.0)

    def gradient(p):
        return 3.0 * np.cos(3.0 * p) * weights + 0.4 * p**3

    lr, beta1, beta2, eps = 2e-3, 0.9, 0.999, 1e-8
    start = np.random.default_rng(5).normal(size=7)

    expected, adam = start.copy(), PlainAdam(lr, beta1, beta2, eps)
    for _ in range(40):
        expected = adam.step(expected, gradient(expected))

    flat = EuclideanManifold(start.shape)
    points = jnp.asarray(start)
    moments = _zero_moments(points)
    for t in range(1, 41):
        rgrad = flat.project(points, jnp.asarray(gradient(np.asarray(points))))
        points, *moments = _adam_update(
            flat,
            points,
            rgrad,
            *moments,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            bc1=1 - beta1**t,
            bc2=1 - beta2**t,
        )
    assert points.dtype == jnp.float64
    np.testing.assert_allclose(points, expected, rtol=1e-13)
    assert np.abs(expected - start).max() > 0.05  # the run moved
