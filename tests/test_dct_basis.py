"""Tests for DCTBasis (Approach B): 1D macro-gate basis init at canonical DCT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import dct

import pdft


def _rand_pic_real(m: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2**m, 2**n)).astype(np.complex128)


def test_dct_basis_construction():
    b = pdft.DCTBasis(m=3, n=3, init_dct=True)
    assert b.image_size == (8, 8)
    assert b.m == 3 and b.n == 3
    assert len(b.tensors) == 2  # one matrix per dim


def test_dct_basis_rejects_zero_dim():
    with pytest.raises(ValueError, match=">= 1"):
        pdft.DCTBasis(m=0, n=1)


def test_dct_basis_init_tensors_match_canonical_dct():
    """At init_dct=True, tensors[0] and tensors[1] are the canonical DCT-II 8×8."""
    b = pdft.DCTBasis(m=3, n=3, init_dct=True)
    # Canonical orthonormal DCT-II 8x8
    n = 8
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n)
    M = np.cos(np.pi * (2 * j + 1) * k / (2 * n))
    M[0, :] *= 1.0 / np.sqrt(n)
    M[1:, :] *= np.sqrt(2.0 / n)
    expected = jnp.asarray(M, dtype=jnp.complex128)
    assert jnp.allclose(b.tensors[0], expected, atol=1e-12)
    assert jnp.allclose(b.tensors[1], expected, atol=1e-12)


def test_dct_basis_init_eye_when_disabled():
    """init_dct=False starts at identity (Walsh-Hadamard fallback would also be valid)."""
    b = pdft.DCTBasis(m=3, n=3, init_dct=False)
    eye_8 = jnp.eye(8, dtype=jnp.complex128)
    assert jnp.allclose(b.tensors[0], eye_8, atol=1e-12)
    assert jnp.allclose(b.tensors[1], eye_8, atol=1e-12)


def test_dct_basis_forward_matches_scipy_dct():
    """At init, forward_transform(pic) must equal scipy's dct-II of the input
    (along both axes), to numerical precision."""
    b = pdft.DCTBasis(m=3, n=3, init_dct=True)
    pic = jnp.asarray(_rand_pic_real(3, 3, seed=11))

    # scipy DCT-II is along an axis; orthonormal flag matches our init.
    expected = jnp.asarray(
        dct(dct(np.asarray(pic).real, type=2, axis=0, norm="ortho"), type=2, axis=1, norm="ortho"),
        dtype=jnp.complex128,
    )
    out = b.forward_transform(pic)
    assert jnp.allclose(out, expected, atol=1e-10)


def test_dct_basis_round_trip():
    b = pdft.DCTBasis(m=3, n=3, init_dct=True)
    pic = jnp.asarray(_rand_pic_real(3, 3, seed=22))
    fwd = b.forward_transform(pic)
    rec = b.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


def test_dct_basis_pytree_round_trip():
    b = pdft.DCTBasis(m=3, n=3, init_dct=True)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    b2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(b2, pdft.DCTBasis)
    pic = jnp.asarray(_rand_pic_real(3, 3, seed=33))
    assert jnp.allclose(b.forward_transform(pic), b2.forward_transform(pic), atol=1e-12)


def test_dct_basis_trains_via_train_basis_batched():
    """train_basis_batched runs end-to-end against DCTBasis."""
    b = pdft.DCTBasis(m=2, n=2, init_dct=True)
    rng = np.random.default_rng(7)
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
        lr_peak=0.001,
        lr_final=0.0001,
        max_grad_norm=1.0,
        shuffle=False,
        seed=42,
    )
    losses = list(res.loss_history)
    assert len(losses) > 0
    head, tail = losses[0], float(np.mean(losses[-max(1, len(losses) // 4) :]))
    assert tail < head * 2.0


def test_dct_basis_block_wrapped():
    inner = pdft.DCTBasis(m=3, n=3, init_dct=True)
    blocked = pdft.BlockedBasis(inner=inner, block_log_m=2, block_log_n=2)
    pic = jnp.asarray(_rand_pic_real(blocked.m, blocked.n, seed=44))
    fwd = blocked.forward_transform(pic)
    rec = blocked.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)
