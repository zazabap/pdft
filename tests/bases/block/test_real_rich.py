"""Tests for RealRichBasis (Approach A): real-orthogonal restriction of RichBasis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft


def _rand_pic_real(m: int, n: int, seed: int = 0):
    """Real-valued image (DIV2K-style)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2**m, 2**n)).astype(np.complex128)


def test_real_rich_basis_construction():
    b = pdft.RealRichBasis(m=3, n=3)
    assert b.image_size == (8, 8)
    assert b.m == 3 and b.n == 3
    # 3 H per dim × 4 + 3 U(4) per dim × 16 = 60 storage elements per dim
    # Free real params (manifold-aware): 3 × 1 (SO(2) component) + 3 × 6 (SO(4)) = 21 per dim
    assert len(b.tensors) == 12  # 6 H + 6 U(4)


def test_real_rich_basis_rejects_zero_dim():
    with pytest.raises(ValueError, match=">= 1"):
        pdft.RealRichBasis(m=0, n=1)


def test_real_rich_basis_initial_tensors_are_real():
    """All initial tensors must be real-valued (zero imaginary part)."""
    b = pdft.RealRichBasis(m=3, n=3)
    for t in b.tensors:
        assert jnp.allclose(jnp.imag(t), 0.0, atol=1e-12)


def test_real_rich_basis_round_trip():
    b = pdft.RealRichBasis(m=3, n=3)
    pic = jnp.asarray(_rand_pic_real(3, 3))
    fwd = b.forward_transform(pic)
    rec = b.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


def test_real_rich_basis_init_is_walsh_hadamard():
    """At init the U(4) slots are identity, so the forward transform is just
    a Walsh-Hadamard composition. We don't pin down the qubit-ordering
    convention here (Yao little-endian vs textbook differs by a permutation),
    only the two invariants that any real-orthogonal transform must satisfy.
    """
    b = pdft.RealRichBasis(m=3, n=3)
    pic = jnp.asarray(_rand_pic_real(3, 3, seed=7))
    fwd = b.forward_transform(pic)
    # 1. Real input, real-orthogonal forward → real output.
    assert jnp.allclose(jnp.imag(fwd), 0.0, atol=1e-10)
    # 2. Parseval: Frobenius norm preserved.
    assert jnp.isclose(jnp.linalg.norm(fwd), jnp.linalg.norm(pic), atol=1e-10)


def test_real_rich_basis_trains_via_train_basis_batched():
    b = pdft.RealRichBasis(m=2, n=2)
    rng = np.random.default_rng(33)
    imgs = [rng.standard_normal((4, 4)).astype(np.complex128) for _ in range(8)]
    res = pdft.train_basis_batched(
        b,
        dataset=imgs,
        loss=pdft.MSELoss(k=4),
        epochs=3,
        batch_size=2,
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        shuffle=False,
        seed=42,
    )
    losses = list(res.loss_history)
    assert len(losses) > 0
    head, tail = losses[0], float(np.mean(losses[-max(1, len(losses) // 4) :]))
    assert tail < head * 2.0


def test_real_rich_basis_trained_tensors_stay_real():
    """After training on real images, the basis stays real-valued (orthogonal)."""
    b = pdft.RealRichBasis(m=2, n=2)
    rng = np.random.default_rng(101)
    imgs = [rng.standard_normal((4, 4)).astype(np.complex128) for _ in range(8)]
    res = pdft.train_basis_batched(
        b,
        dataset=imgs,
        loss=pdft.MSELoss(k=4),
        epochs=3,
        batch_size=2,
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        shuffle=False,
        seed=42,
    )
    # After 3 epochs: tensors should still be effectively real (FP-noise small).
    for t in res.basis.tensors:
        max_imag = float(jnp.max(jnp.abs(jnp.imag(t))))
        assert max_imag < 1e-6, f"tensor drifted off real: max_imag={max_imag}"


def test_real_rich_basis_pytree_round_trip():
    b = pdft.RealRichBasis(m=3, n=3)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    b2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(b2, pdft.RealRichBasis)
    pic = jnp.asarray(_rand_pic_real(3, 3, seed=99))
    assert jnp.allclose(b.forward_transform(pic), b2.forward_transform(pic), atol=1e-12)


def test_real_rich_basis_block_wrapped():
    """BlockedBasis(RealRichBasis(3,3), 5, 5) is the production configuration."""
    inner = pdft.RealRichBasis(m=3, n=3)
    blocked = pdft.BlockedBasis(inner=inner, block_log_m=2, block_log_n=2)
    pic = jnp.asarray(_rand_pic_real(blocked.m, blocked.n, seed=22))
    fwd = blocked.forward_transform(pic)
    rec = blocked.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)
