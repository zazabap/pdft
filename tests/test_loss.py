import jax.numpy as jnp
import pytest

from pdft.loss import L1Norm, MSELoss, topk_truncate


def test_l1norm_is_stateless():
    a, b = L1Norm(), L1Norm()
    assert a == b


def test_mseloss_requires_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        MSELoss(k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        MSELoss(k=-1)


def test_mseloss_stores_k():
    assert MSELoss(k=5).k == 5


def test_topk_truncate_k_equals_length_is_identity():
    x = jnp.array([[3.0 + 0j, -1.0, 2.0, 0.5]])
    out = topk_truncate(x, k=4)
    assert jnp.allclose(out, x)


def test_topk_truncate_zero_k_zeros_everything():
    x = jnp.array([[3.0 + 0j, -1.0, 2.0, 0.5]])
    out = topk_truncate(x, k=0)
    assert jnp.allclose(out, jnp.zeros_like(x))


def test_topk_truncate_keeps_largest_magnitudes():
    x = jnp.array([[1.0 + 0j, -3.0, 2.0, 0.1]])
    out = topk_truncate(x, k=2)
    expected = jnp.array([[0.0 + 0j, -3.0, 2.0, 0.0]])
    assert jnp.allclose(out, expected)


def test_topk_truncate_k_larger_than_length_clamps():
    x = jnp.array([[1.0 + 0j, 2.0]])
    out = topk_truncate(x, k=10)
    assert jnp.allclose(out, x)
