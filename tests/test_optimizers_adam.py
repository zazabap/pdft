"""RiemannianAdam unit + integration tests."""

import jax
import jax.numpy as jnp

from pdft.optimizers import RiemannianAdam, optimize


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

    from pdft.basis import QFTBasis
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
