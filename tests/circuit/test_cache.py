import jax.numpy as jnp

from pdft.circuit.cache import optimize_code_cached


def test_optimize_code_cached_returns_callable():
    fn = optimize_code_cached("ij,jk->ik", (3, 4), (4, 5))
    A = jnp.ones((3, 4))
    B = jnp.ones((4, 5))
    out = fn(A, B)
    assert out.shape == (3, 5)


def test_optimize_code_cached_reuses_entry():
    fn1 = optimize_code_cached("ij,jk->ik", (3, 4), (4, 5))
    fn2 = optimize_code_cached("ij,jk->ik", (3, 4), (4, 5))
    assert fn1 is fn2  # cache returns identical object


def test_optimize_code_cached_distinct_shapes_distinct_entries():
    fn1 = optimize_code_cached("ij,jk->ik", (3, 4), (4, 5))
    fn2 = optimize_code_cached("ij,jk->ik", (7, 8), (8, 9))
    assert fn1 is not fn2
