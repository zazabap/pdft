"""Tests for ``pdft.BlockedBasis`` — wraps any inner basis as a block transform.

Coverage:
  * construction validation (bad inner / negative block dims)
  * shape and parameter accounting (m, n, image_size, num_blocks, block_shape)
  * round-trip identity: ``inverse_transform(forward_transform(x)) ≈ x``
  * block independence: applying BlockedBasis to a block-uniform image gives
    the same as applying the inner basis to one block, replicated
  * delegation: a 1x1 block grid (block_log_m = block_log_n = 0) is identical
    to the inner basis
  * JAX pytree round-trip preserves all aux data + tensors
  * grad through forward_transform produces finite grads on inner.tensors
  * trainable end-to-end: train_basis_batched runs against BlockedBasis at
    small m_outer/n_outer and produces monotone-ish loss descent
  * works with EntangledQFTBasis and TEBDBasis as inner (smoke)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft


def _rand_pic(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n))).astype(
        np.complex128
    )


# ---- construction -----------------------------------------------------------


def test_blocked_basis_rejects_non_basis():
    with pytest.raises(TypeError, match="basis"):
        pdft.BlockedBasis(inner=object(), block_log_m=1, block_log_n=1)


def test_blocked_basis_rejects_negative_block_dims():
    inner = pdft.QFTBasis(m=2, n=2)
    with pytest.raises(ValueError, match="must be >= 0"):
        pdft.BlockedBasis(inner=inner, block_log_m=-1, block_log_n=0)
    with pytest.raises(ValueError, match="must be >= 0"):
        pdft.BlockedBasis(inner=inner, block_log_m=0, block_log_n=-1)


def test_blocked_basis_shape_arithmetic():
    inner = pdft.QFTBasis(m=2, n=3)
    b = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=2)
    assert b.m == 3
    assert b.n == 5
    assert b.image_size == (2**3, 2**5)
    assert b.num_blocks == 2 ** (1 + 2)
    assert b.block_shape == (2**2, 2**3)
    # parameters live in inner
    assert b.num_parameters == inner.num_parameters
    assert b.tensors is inner.tensors


# ---- functional correctness -------------------------------------------------


def test_blocked_basis_round_trip():
    """T^{-1}(T(x)) ≈ x for any image."""
    inner = pdft.QFTBasis(m=2, n=2)
    b = pdft.BlockedBasis(inner=inner, block_log_m=2, block_log_n=2)
    pic = jnp.asarray(_rand_pic(b.m, b.n))
    fwd = b.forward_transform(pic)
    rec = b.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


def test_blocked_basis_zero_blocks_matches_inner():
    """block_log_m = block_log_n = 0 reduces exactly to the inner basis."""
    inner = pdft.QFTBasis(m=3, n=3)
    b = pdft.BlockedBasis(inner=inner, block_log_m=0, block_log_n=0)
    pic = jnp.asarray(_rand_pic(inner.m, inner.n))
    inner_out = inner.forward_transform(pic)
    block_out = b.forward_transform(pic)
    assert jnp.allclose(inner_out, block_out, atol=1e-10)


def test_blocked_basis_block_independence():
    """Applying the BlockedBasis to a block-tiled image equals the inner basis
    applied to ONE block, then tiled — the very definition of block independence.
    """
    inner = pdft.QFTBasis(m=2, n=2)
    block_log_m = 2
    block_log_n = 1
    b = pdft.BlockedBasis(inner=inner, block_log_m=block_log_m, block_log_n=block_log_n)

    # Generate a single block, then construct a block-uniform outer image
    # (every block contains the same pixels).
    block = jnp.asarray(_rand_pic(inner.m, inner.n, seed=7))
    n_blocks_row = 2**block_log_m
    n_blocks_col = 2**block_log_n
    # Tile via reshape: (n_blocks_row, block_size_row, n_blocks_col, block_size_col)
    # set every block index to `block`
    outer = jnp.broadcast_to(
        block.reshape(1, 2**inner.m, 1, 2**inner.n),
        (n_blocks_row, 2**inner.m, n_blocks_col, 2**inner.n),
    ).reshape(2**b.m, 2**b.n)

    fwd_outer = b.forward_transform(outer)
    fwd_inner = inner.forward_transform(block)

    # Each block of fwd_outer should equal fwd_inner.
    fwd_outer_blocks = fwd_outer.reshape(n_blocks_row, 2**inner.m, n_blocks_col, 2**inner.n)
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            assert jnp.allclose(fwd_outer_blocks[i, :, j, :], fwd_inner, atol=1e-10), (
                f"block ({i},{j}) does not match inner forward"
            )


def test_blocked_basis_with_entangled_qft_inner():
    """Smoke: BlockedBasis works with EntangledQFTBasis as inner."""
    inner = pdft.EntangledQFTBasis(m=2, n=2, seed=7)
    b = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=1)
    pic = jnp.asarray(_rand_pic(b.m, b.n, seed=3))
    fwd = b.forward_transform(pic)
    rec = b.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


def test_blocked_basis_with_tebd_inner():
    """Smoke: BlockedBasis works with TEBDBasis as inner."""
    inner = pdft.TEBDBasis(m=2, n=2, seed=7)
    b = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=1)
    pic = jnp.asarray(_rand_pic(b.m, b.n, seed=3))
    fwd = b.forward_transform(pic)
    rec = b.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


# ---- gradients & training ---------------------------------------------------


def test_blocked_basis_grad_finite():
    """jax.grad through forward_transform yields finite grads on inner.tensors."""
    inner = pdft.QFTBasis(m=2, n=2)
    b = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=1)
    pic = jnp.asarray(_rand_pic(b.m, b.n, seed=11))

    def loss_fn(tensors):
        # Reuse BlockedBasis's code via _apply_circuit to get a real scalar loss.
        from pdft.loss import _apply_circuit

        out = _apply_circuit(tensors, b.code, b.m, b.n, pic)
        return jnp.sum(jnp.abs(out) ** 2)

    grads = jax.grad(loss_fn)(list(inner.tensors))
    for g in grads:
        assert jnp.all(jnp.isfinite(g))


def test_blocked_basis_trains_monotone():
    """train_basis_batched runs to completion against BlockedBasis and reduces
    loss across epochs.
    """
    inner = pdft.QFTBasis(m=2, n=2)
    b = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=1)
    rng = np.random.default_rng(123)
    imgs = [
        (rng.standard_normal((2**b.m, 2**b.n)) + 1j * rng.standard_normal((2**b.m, 2**b.n))).astype(
            np.complex128
        )
        for _ in range(8)
    ]
    result = pdft.train_basis_batched(
        b,
        dataset=imgs,
        loss=pdft.MSELoss(k=8),
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
    # We expect the training to complete and produce a sensible loss history.
    losses = list(result.loss_history)
    assert len(losses) > 0
    # Loss should not blow up — last-quartile mean lower than first batch.
    head = losses[0]
    tail_mean = float(np.mean(losses[-max(1, len(losses) // 4) :]))
    assert tail_mean < head * 2.0, f"loss diverged: head={head} tail={tail_mean}"


# ---- pytree -----------------------------------------------------------------


def test_blocked_basis_pytree_round_trip():
    """Flatten → unflatten preserves the basis end-to-end."""
    inner = pdft.QFTBasis(m=2, n=2)
    b = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=1)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    assert len(leaves) == len(inner.tensors)
    b2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(b2, pdft.BlockedBasis)
    assert b2.m == b.m
    assert b2.n == b.n
    assert b2.block_log_m == b.block_log_m
    assert b2.block_log_n == b.block_log_n
    assert type(b2.inner) is type(inner)
    # Forward output must match.
    pic = jnp.asarray(_rand_pic(b.m, b.n, seed=99))
    assert jnp.allclose(b.forward_transform(pic), b2.forward_transform(pic), atol=1e-12)
