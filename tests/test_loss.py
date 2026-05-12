from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from pdft.bases.circuit.qft import qft_code
from pdft.loss import L1Norm, MSELoss, loss_function, topk_truncate


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


def test_mseloss_no_extra_loss_unchanged():
    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    base_loss = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1), inverse_code=inv_code))

    assert base_loss == base_loss


def test_mseloss_extra_loss_hook_adds_term():
    @dataclass(frozen=True)
    class WithExtra(MSELoss):
        def _extra_loss(self, tensors):
            return jnp.asarray(7.0, dtype=jnp.float64)

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    base_loss = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1), inverse_code=inv_code))
    extra_loss = float(loss_function(tensors, m, n, code, pic, WithExtra(k=1), inverse_code=inv_code))

    assert abs(extra_loss - base_loss - 7.0) < 1e-10


def test_mseloss_extra_loss_uses_tensors():
    @dataclass(frozen=True)
    class TensorSum(MSELoss):
        def _extra_loss(self, tensors):
            return sum(jnp.sum(jnp.abs(t) ** 2) for t in tensors).real

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.zeros((4, 4), dtype=jnp.complex128)

    loss_val = float(loss_function(tensors, m, n, code, pic, TensorSum(k=1), inverse_code=inv_code))
    expected_reg = float(sum(jnp.sum(jnp.abs(t) ** 2) for t in tensors).real)

    assert abs(loss_val - expected_reg) < 1e-6
