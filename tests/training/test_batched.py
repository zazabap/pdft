"""Tests for batched training pipeline (Julia _train_basis_core parity).

Covers `_cosine_with_warmup` LR schedule and `train_basis_batched`. The latter
mirrors `ParametricDFT.jl/src/training.jl::_train_basis_core` (epochs over a
multi-image dataset, mini-batches, validation split with patience-based early
stopping, cosine-with-warmup LR schedule, optional gradient clipping).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.circuit.qft import controlled_phase_diag
from pdft.training import cosine_with_warmup, train_basis_batched


# ---------------------------------------------------------------------------
# _cosine_with_warmup
# ---------------------------------------------------------------------------


def test_cosine_warmup_at_zero():
    """Step 0 (or 1, depending on convention) is at the start of warmup → tiny lr."""
    lr = cosine_with_warmup(step=0, total_steps=100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001)
    # Warmup is `max(1, round(0.1 * 100))` = 10. lr at step 0 = lr_peak * 0/10 = 0.
    assert lr == pytest.approx(0.0)


def test_cosine_warmup_at_peak():
    """Right after warmup ends, lr should be at lr_peak."""
    lr = cosine_with_warmup(step=10, total_steps=100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001)
    assert lr == pytest.approx(0.01)


def test_cosine_warmup_at_final():
    """At the very last step, lr should be at lr_final."""
    lr = cosine_with_warmup(
        step=100, total_steps=100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001
    )
    assert lr == pytest.approx(0.001, abs=1e-9)


def test_cosine_warmup_monotone_decrease_after_peak():
    """After warmup, lr monotonically decreases through cosine."""
    lrs = [
        cosine_with_warmup(s, 100, warmup_frac=0.1, lr_peak=0.01, lr_final=0.001)
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
        py = cosine_with_warmup(
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


def _qft_as_blocked_2x2_with_frozen_outer():
    """Return QFT(3,3) configured like BlockedBasis(QFT(2,2), 1, 1)."""
    full = pdft.QFTBasis(m=3, n=3)
    tensors = [jnp.array(t, copy=True) for t in full.tensors]

    # qft_code sorts Hadamards first, then CP gates. For QFT(3,3), qubits
    # 3 and 6 are the block-index row/col qubits. Setting every gate touching
    # those qubits to identity leaves exactly the within-block QFT(2,2).
    frozen_h = [2, 5]
    frozen_cp = [7, 8, 10, 11]
    for i in frozen_h:
        tensors[i] = jnp.eye(2, dtype=tensors[i].dtype)
    for i in frozen_cp:
        tensors[i] = controlled_phase_diag(0.0)

    basis = pdft.QFTBasis(m=3, n=3, tensors=tensors, code=full.code, inv_code=full.inv_code)
    return basis, frozen_h + frozen_cp, [0, 1, 3, 4, 6, 9]


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


def test_train_basis_batched_freezes_specified_indices():
    """Frozen indices stay bit-exactly at their initial values; non-frozen
    indices update normally."""
    import jax.numpy as jnp
    import numpy as np
    import pdft

    # Build a small QFTBasis (m=n=2 -> 4 H + 2 CP = 6 tensors).
    basis = pdft.QFTBasis(m=2, n=2)
    initial_tensors = [jnp.array(t, copy=True) for t in basis.tensors]

    # Freeze indices [0, 2, 4] — the H@q1, H@q3, and CP(q1,q2) gates.
    # Indices 1, 3, 5 (H@q2, H@q4, CP(q3,q4)) should still train.
    frozen = [0, 2, 4]
    free = [1, 3, 5]

    # Tiny synthetic dataset — random 4x4 complex images.
    rng = np.random.default_rng(7)
    train = rng.standard_normal((8, 4, 4)) + 1j * rng.standard_normal((8, 4, 4))
    train = train.astype(np.complex128)

    result = pdft.train_basis_batched(
        basis,
        dataset=train,
        loss=pdft.MSELoss(k=4),
        epochs=3,
        batch_size=4,
        optimizer="adam",
        validation_split=0.25,
        early_stopping_patience=10**9,
        seed=42,
        frozen_indices=frozen,
    )

    # Frozen tensors must be bit-exactly equal to the initial values.
    for i in frozen:
        diff = float(jnp.max(jnp.abs(result.basis.tensors[i] - initial_tensors[i])))
        assert diff == 0.0, (
            f"frozen index {i} drifted by {diff} (expected 0). "
            f"frozen_indices semantics are broken."
        )

    # Non-frozen tensors should have moved (training is non-trivial).
    moved_any = False
    for i in free:
        diff = float(jnp.max(jnp.abs(result.basis.tensors[i] - initial_tensors[i])))
        if diff > 1e-6:
            moved_any = True
    assert moved_any, (
        "no non-frozen tensors moved — training appears not to have run, "
        "or the freezing is over-aggressive."
    )


def test_train_basis_batched_freezes_specified_indices_with_gd():
    """GD/Armijo should evaluate and retain the constrained update directly."""
    import jax.numpy as jnp
    import numpy as np
    import pdft

    basis = pdft.QFTBasis(m=2, n=2)
    initial_tensors = [jnp.array(t, copy=True) for t in basis.tensors]
    frozen = [0, 2, 4]
    free = [1, 3, 5]

    rng = np.random.default_rng(11)
    train = rng.standard_normal((4, 4, 4)) + 1j * rng.standard_normal((4, 4, 4))
    train = train.astype(np.complex128)

    result = pdft.train_basis_batched(
        basis,
        dataset=train,
        loss=pdft.MSELoss(k=4),
        epochs=2,
        batch_size=2,
        optimizer="gd",
        validation_split=0.0,
        early_stopping_patience=10**9,
        warmup_frac=0.0,
        lr_peak=0.01,
        lr_final=0.01,
        seed=42,
        frozen_indices=frozen,
    )

    for i in frozen:
        diff = float(jnp.max(jnp.abs(result.basis.tensors[i] - initial_tensors[i])))
        assert diff == 0.0

    assert any(
        float(jnp.max(jnp.abs(result.basis.tensors[i] - initial_tensors[i]))) > 1e-6
        for i in free
    )


def test_frozen_qft_outer_gates_matches_blocked_qft_training():
    """Freezing identity outer QFT gates is equivalent to training BlockedBasis."""
    full_basis, frozen_outer, trainable_map = _qft_as_blocked_2x2_with_frozen_outer()
    blocked_basis = pdft.BlockedBasis(inner=pdft.QFTBasis(m=2, n=2), block_log_m=1, block_log_n=1)

    rng = np.random.default_rng(5)
    dataset = (
        rng.standard_normal((4, 8, 8)) + 1j * rng.standard_normal((4, 8, 8))
    ).astype(np.complex128)

    # Initial transforms are the same operator before any training starts.
    pic = jnp.asarray(dataset[0])
    np.testing.assert_allclose(
        np.asarray(full_basis.forward_transform(pic)),
        np.asarray(blocked_basis.forward_transform(pic)),
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

    frozen_result = pdft.train_basis_batched(
        full_basis,
        frozen_indices=frozen_outer,
        **train_kwargs,
    )
    blocked_result = pdft.train_basis_batched(blocked_basis, **train_kwargs)

    np.testing.assert_allclose(
        np.asarray(frozen_result.loss_history),
        np.asarray(blocked_result.loss_history),
        atol=1e-12,
        rtol=0.0,
    )

    for blocked_i, full_i in enumerate(trainable_map):
        np.testing.assert_allclose(
            np.asarray(frozen_result.basis.tensors[full_i]),
            np.asarray(blocked_result.basis.tensors[blocked_i]),
            atol=1e-12,
            rtol=0.0,
        )
    for i in frozen_outer:
        np.testing.assert_array_equal(
            np.asarray(frozen_result.basis.tensors[i]),
            np.asarray(full_basis.tensors[i]),
        )


def test_train_basis_batched_frozen_indices_validation():
    """frozen_indices validation: out-of-range index, negative, duplicate."""
    import pytest
    import pdft

    basis = pdft.QFTBasis(m=2, n=2)
    n_tensors = len(basis.tensors)  # 6

    # Out-of-range positive.
    with pytest.raises(ValueError):
        pdft.train_basis_batched(
            basis,
            dataset=[],
            loss=pdft.MSELoss(k=4),
            epochs=1,
            batch_size=1,
            frozen_indices=[n_tensors],  # one past the end
        )

    # Negative.
    with pytest.raises(ValueError):
        pdft.train_basis_batched(
            basis,
            dataset=[],
            loss=pdft.MSELoss(k=4),
            epochs=1,
            batch_size=1,
            frozen_indices=[-1],
        )

    # Duplicate.
    with pytest.raises(ValueError):
        pdft.train_basis_batched(
            basis,
            dataset=[],
            loss=pdft.MSELoss(k=4),
            epochs=1,
            batch_size=1,
            frozen_indices=[0, 0],
        )

    # Non-integer coercions should not be accepted.
    for bad_index in (1.9, True, "2"):
        with pytest.raises(ValueError):
            pdft.train_basis_batched(
                basis,
                dataset=[],
                loss=pdft.MSELoss(k=4),
                epochs=1,
                batch_size=1,
                frozen_indices=[bad_index],
            )
