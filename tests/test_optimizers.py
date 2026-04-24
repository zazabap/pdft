import jax
import jax.numpy as jnp

from pdft.optimizers import RiemannianGD, optimize


def test_riemannian_gd_defaults_match_upstream():
    opt = RiemannianGD()
    assert opt.lr == 0.01
    assert opt.armijo_c == 1e-4
    assert opt.armijo_tau == 0.5
    assert opt.max_ls_steps == 10
    assert opt.max_grad_norm is None


def _random_unitary_tensors(d: int, count: int, key=jax.random.PRNGKey(0)) -> list:
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (count, d, d)) + 1j * jax.random.normal(k2, (count, d, d))
    Q, _ = jnp.linalg.qr(A)
    return [Q[i].astype(jnp.complex128) for i in range(count)]


def test_optimize_preserves_unitarity_and_monotonicity():
    tensors = _random_unitary_tensors(d=4, count=3)

    # Toy loss: sum of real trace; grad is simple and deterministic
    def loss_fn(ts):
        return sum(jnp.real(jnp.trace(t)) for t in ts)

    grad_fn = jax.grad(loss_fn)
    opt = RiemannianGD(lr=0.1)
    final, trace = optimize(
        opt, tensors, loss_fn, grad_fn, max_iter=50, tol=1e-10, record_loss=True
    )

    for t in final:
        d = t.shape[0]
        assert jnp.allclose(t @ jnp.conj(t).T, jnp.eye(d), atol=1e-6)

    losses = jnp.array(trace)
    assert jnp.all(jnp.diff(losses) <= 1e-8)


def test_optimize_rejects_bad_params():
    import pytest

    tensors = _random_unitary_tensors(d=2, count=1)
    loss_fn = lambda ts: jnp.real(jnp.trace(ts[0]))
    grad_fn = jax.grad(loss_fn)

    with pytest.raises(ValueError):
        optimize(RiemannianGD(lr=0.01), tensors, loss_fn, grad_fn, max_iter=0, tol=1e-6)
    with pytest.raises(ValueError):
        optimize(RiemannianGD(lr=0.0), tensors, loss_fn, grad_fn, max_iter=10, tol=1e-6)
