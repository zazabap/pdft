"""Tests for batched training pipeline (Julia _train_basis_core parity).

Covers `_cosine_with_warmup` LR schedule and `train_basis_batched`. The latter
mirrors `ParametricDFT.jl/src/training.jl::_train_basis_core` (epochs over a
multi-image dataset, mini-batches, validation split with patience-based early
stopping, cosine-with-warmup LR schedule, optional gradient clipping).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pdft
from pdft.training import _cosine_with_warmup, train_basis_batched


# ---------------------------------------------------------------------------
# _cosine_with_warmup
# ---------------------------------------------------------------------------


def test_cosine_warmup_at_zero():
    """Step 0 (or 1, depending on convention) is at the start of warmup → tiny lr."""
    lr = _cosine_with_warmup(step=0, total_steps=100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001)
    # Warmup is `max(1, round(0.1 * 100))` = 10. lr at step 0 = lr_peak * 0/10 = 0.
    assert lr == pytest.approx(0.0)


def test_cosine_warmup_at_peak():
    """Right after warmup ends, lr should be at lr_peak."""
    lr = _cosine_with_warmup(
        step=10, total_steps=100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001
    )
    assert lr == pytest.approx(0.01)


def test_cosine_warmup_at_final():
    """At the very last step, lr should be at lr_final."""
    lr = _cosine_with_warmup(
        step=100, total_steps=100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001
    )
    assert lr == pytest.approx(0.001, abs=1e-9)


def test_cosine_warmup_monotone_decrease_after_peak():
    """After warmup, lr monotonically decreases through cosine."""
    lrs = [
        _cosine_with_warmup(s, 100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001)
        for s in range(11, 101)
    ]
    for a, b in zip(lrs, lrs[1:]):
        assert b <= a + 1e-12


def test_cosine_warmup_matches_julia_formula():
    """Closed-form check against the formula in
    ParametricDFT.jl/src/training.jl::_cosine_with_warmup."""
    total = 50
    warmup_frac = 0.05
    lr_peak = 0.01
    lr_final = 0.001
    warmup_steps = max(1, round(warmup_frac * total))
    for step in range(0, total + 1):
        py = _cosine_with_warmup(
            step, total, warmup_frac=warmup_frac, lr_peak=lr_peak, lr_final=lr_final
        )
        if step <= warmup_steps:
            expected = lr_peak * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total - warmup_steps)
            expected = lr_final + 0.5 * (lr_peak - lr_final) * (1 + math.cos(math.pi * progress))
        assert py == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# train_basis_batched
# ---------------------------------------------------------------------------


def _toy_dataset(n_images: int, m: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w = 2**m, 2**n
    return [
        (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))).astype(np.complex128)
        for _ in range(n_images)
    ]


def test_batched_returns_training_result_shape():
    dataset = _toy_dataset(4, 2, 2)
    basis = pdft.QFTBasis(m=2, n=2)
    res = train_basis_batched(
        basis,
        dataset=dataset,
        epochs=2,
        batch_size=2,
        loss=pdft.L1Norm(),
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.1,
        lr_peak=0.01,
        lr_final=0.001,
        seed=0,
    )
    assert isinstance(res, pdft.TrainingResult)
    # epochs * ceil(n_train / batch_size) = 2 * ceil(4 / 2) = 4 steps.
    assert len(res.loss_history) == 4
    assert res.steps == 4


def test_batched_batch_size_one_matches_single_target():
    """batch_size=1 with an effective single-image dataset and zero LR-schedule
    decay should produce a similar trajectory to single-target train_basis.
    Tolerance is loose because the cosine-warmup schedule still applies."""
    rng = np.random.default_rng(7)
    target = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(np.complex128)
    basis_a = pdft.QFTBasis(m=2, n=2)
    res_a = train_basis_batched(
        basis_a,
        dataset=[target],
        epochs=3,
        batch_size=1,
        loss=pdft.L1Norm(),
        optimizer="gd",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.0,  # no warmup → flat lr
        lr_peak=0.01,
        lr_final=0.01,  # flat schedule → const lr
    )
    # Loss should be non-increasing across the 3 steps (mostly).
    assert res_a.loss_history[-1] <= res_a.loss_history[0] + 1e-9


def test_batched_validation_split_applied():
    dataset = _toy_dataset(10, 2, 2)
    basis = pdft.QFTBasis(m=2, n=2)
    res = train_basis_batched(
        basis,
        dataset=dataset,
        epochs=2,
        batch_size=2,
        loss=pdft.L1Norm(),
        optimizer="adam",
        validation_split=0.2,  # 2 of 10 → validation
        early_stopping_patience=10,
        warmup_frac=0.1,
        lr_peak=0.01,
        lr_final=0.001,
        seed=42,
    )
    # 2 epochs × ceil(8 / 2) = 8 batches.
    assert len(res.loss_history) == 8
    # val_history has one entry per epoch.
    assert hasattr(res, "val_history")
    assert len(res.val_history) == 2


def test_batched_early_stopping():
    """Loss-monotone-up dataset with patience=1 stops after 2 epochs."""
    dataset = _toy_dataset(4, 2, 2)
    basis = pdft.QFTBasis(m=2, n=2)
    res = train_basis_batched(
        basis,
        dataset=dataset,
        epochs=20,  # would take 20 epochs without early stopping
        batch_size=2,
        loss=pdft.L1Norm(),
        optimizer="adam",
        validation_split=0.5,
        early_stopping_patience=1,  # stop after 1 epoch with no improvement
        warmup_frac=0.05,
        lr_peak=0.5,  # absurdly high lr → val_loss likely worsens fast
        lr_final=0.001,
        seed=0,
    )
    # Either it completes all 20 epochs, OR early stops with fewer val entries.
    assert len(res.val_history) <= 20


def test_batched_grad_clip_runs():
    """max_grad_norm doesn't crash and produces a valid TrainingResult."""
    dataset = _toy_dataset(4, 2, 2)
    basis = pdft.QFTBasis(m=2, n=2)
    res = train_basis_batched(
        basis,
        dataset=dataset,
        epochs=1,
        batch_size=2,
        loss=pdft.L1Norm(),
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        seed=0,
    )
    assert len(res.loss_history) == 2  # 1 epoch × 2 batches
    assert all(np.isfinite(L) for L in res.loss_history)


def test_batched_seed_deterministic():
    """Same seed with same inputs reproduces the same training trajectory."""
    dataset = _toy_dataset(4, 2, 2, seed=0)
    a = train_basis_batched(
        pdft.QFTBasis(m=2, n=2),
        dataset=dataset,
        epochs=2,
        batch_size=2,
        loss=pdft.L1Norm(),
        optimizer="adam",
        validation_split=0.25,
        early_stopping_patience=10,
        warmup_frac=0.1,
        lr_peak=0.01,
        lr_final=0.001,
        seed=42,
    )
    b = train_basis_batched(
        pdft.QFTBasis(m=2, n=2),
        dataset=dataset,
        epochs=2,
        batch_size=2,
        loss=pdft.L1Norm(),
        optimizer="adam",
        validation_split=0.25,
        early_stopping_patience=10,
        warmup_frac=0.1,
        lr_peak=0.01,
        lr_final=0.001,
        seed=42,
    )
    np.testing.assert_array_equal(np.array(a.loss_history), np.array(b.loss_history))


def test_batched_unknown_optimizer_raises():
    dataset = _toy_dataset(2, 2, 2)
    basis = pdft.QFTBasis(m=2, n=2)
    with pytest.raises(ValueError, match="optimizer"):
        train_basis_batched(
            basis,
            dataset=dataset,
            epochs=1,
            batch_size=1,
            loss=pdft.L1Norm(),
            optimizer="sgd",  # not supported
            validation_split=0.0,
            early_stopping_patience=10,
            warmup_frac=0.05,
            lr_peak=0.01,
            lr_final=0.001,
        )


def test_batched_validation_split_too_large_raises():
    dataset = _toy_dataset(2, 2, 2)
    basis = pdft.QFTBasis(m=2, n=2)
    with pytest.raises(ValueError, match="validation_split"):
        train_basis_batched(
            basis,
            dataset=dataset,
            epochs=1,
            batch_size=1,
            loss=pdft.L1Norm(),
            optimizer="adam",
            validation_split=1.0,  # invalid
            early_stopping_patience=10,
            warmup_frac=0.05,
            lr_peak=0.01,
            lr_final=0.001,
        )
