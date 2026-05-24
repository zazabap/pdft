"""Tests for freeze_as_blocked: outer-gate freezing → BlockedBasis equivalence."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.circuit import freeze_as_blocked


def test_freeze_as_blocked_qft_indices():
    """QFT(3,3) blocked 1+1 freezes exactly the hand-computed outer indices."""
    basis = pdft.QFTBasis(m=3, n=3)
    frozen_basis, frozen = freeze_as_blocked(basis, 1, 1)
    assert frozen == [2, 5, 7, 8, 10, 11]
    # Frozen outer gates were reset to identity: the row/col block-index H
    # gates (sorted indices 2, 5) become eye(2).
    np.testing.assert_allclose(
        np.asarray(frozen_basis.tensors[2]), np.eye(2), atol=1e-12, rtol=0.0
    )


def test_freeze_as_blocked_rich_resets_outer_u4_to_identity():
    """Rich(3,3) blocked 1+1: same outer indices as QFT; outer U4 gates reset
    to the 4x4 identity, leaving 6 trainable inner gates (== BlockedBasis(2,2))."""
    basis = pdft.RichBasis(m=3, n=3)
    frozen_basis, frozen = freeze_as_blocked(basis, 1, 1)
    assert frozen == [2, 5, 7, 8, 10, 11]
    assert len(frozen_basis.tensors) - len(frozen) == 6
    # Frozen U4 slots (the (2,2,2,2)-shaped ones) become eye(4) reshaped.
    eye4 = np.eye(4).reshape(2, 2, 2, 2)
    checked_u4 = False
    for i in frozen:
        t = np.asarray(frozen_basis.tensors[i])
        if t.shape == (2, 2, 2, 2):
            np.testing.assert_allclose(t, eye4, atol=1e-12, rtol=0.0)
            checked_u4 = True
    assert checked_u4, "expected at least one frozen U4 gate"


def test_freeze_as_blocked_rejects_bad_partition():
    basis = pdft.QFTBasis(m=3, n=3)
    with pytest.raises(ValueError):
        freeze_as_blocked(basis, 3, 1)  # m - block_log_m = 0 < 1
    with pytest.raises(ValueError):
        freeze_as_blocked(basis, -1, 1)  # negative


def test_freeze_as_blocked_rejects_unsupported_type():
    basis = pdft.TEBDBasis(m=2, n=2, seed=0)
    with pytest.raises(TypeError):
        freeze_as_blocked(basis, 1, 1)


def _parity_check(basis_factory):
    """freeze_as_blocked(full(3,3)) trained == BlockedBasis(inner(2,2)) trained."""
    full0 = basis_factory(3, 3)
    full, frozen = freeze_as_blocked(full0, 1, 1)
    # freeze_as_blocked's contract: frozen_indices come back ascending. The
    # positional trainable<->inner correspondence asserted below relies on it.
    assert frozen == sorted(frozen)
    blocked = pdft.BlockedBasis(
        inner=basis_factory(2, 2), block_log_m=1, block_log_n=1
    )

    rng = np.random.default_rng(5)
    dataset = (
        rng.standard_normal((4, 8, 8)) + 1j * rng.standard_normal((4, 8, 8))
    ).astype(np.complex128)

    # Initial forward transforms must already be the same operator.
    pic = jnp.asarray(dataset[0])
    np.testing.assert_allclose(
        np.asarray(full.forward_transform(pic)),
        np.asarray(blocked.forward_transform(pic)),
        atol=1e-12,
        rtol=0.0,
    )

    train_kwargs = dict(
        dataset=dataset,
        loss=pdft.MSELoss(k=8),
        epochs=2,
        batch_size=2,
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10**9,
        warmup_frac=0.0,
        lr_peak=0.003,
        lr_final=0.003,
        max_grad_norm=1.0,
        shuffle=False,
        seed=123,
    )

    frozen_result = pdft.train_basis_batched(full, frozen_indices=frozen, **train_kwargs)
    blocked_result = pdft.train_basis_batched(blocked, **train_kwargs)

    np.testing.assert_allclose(
        np.asarray(frozen_result.loss_history),
        np.asarray(blocked_result.loss_history),
        atol=1e-12,
        rtol=0.0,
    )

    # Trainable (non-frozen) tensors, in ascending index order, correspond
    # positionally to BlockedBasis's inner tensors.
    trainable_map = [i for i in range(len(full.tensors)) if i not in set(frozen)]
    assert len(trainable_map) + len(frozen) == len(full.tensors)  # exhaustive split
    assert len(trainable_map) == len(blocked.tensors)
    for blocked_i, full_i in enumerate(trainable_map):
        np.testing.assert_allclose(
            np.asarray(frozen_result.basis.tensors[full_i]),
            np.asarray(blocked_result.basis.tensors[blocked_i]),
            atol=1e-12,
            rtol=0.0,
        )
    # Frozen tensors never moved.
    for i in frozen:
        np.testing.assert_array_equal(
            np.asarray(frozen_result.basis.tensors[i]),
            np.asarray(full.tensors[i]),
        )


def test_freeze_as_blocked_rich_matches_blocked():
    _parity_check(lambda m, n: pdft.RichBasis(m=m, n=n))


def test_freeze_as_blocked_real_rich_matches_blocked():
    _parity_check(lambda m, n: pdft.RealRichBasis(m=m, n=n))
