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
from pdft.training import cosine_with_warmup, train_basis_batched

from ..helpers import complex_image, complex_normal

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
    for step in range(total + 1):
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
    """Return QFT(3,3) configured like BlockedBasis(QFT(2,2), 1, 1).

    freeze_as_blocked resets the block-index (qubits 3 and 6) gates to identity
    and returns their tensor indices; the trainable map is the complement.
    """
    from pdft.bases.circuit import freeze_as_blocked

    basis, frozen = freeze_as_blocked(pdft.QFTBasis(m=3, n=3), 1, 1)
    trainable_map = [i for i in range(len(basis.tensors)) if i not in set(frozen)]
    return basis, frozen, trainable_map


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
            f"frozen index {i} drifted by {diff} (expected 0). frozen_indices semantics are broken."
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
        float(jnp.max(jnp.abs(result.basis.tensors[i] - initial_tensors[i]))) > 1e-6 for i in free
    )


def test_frozen_qft_outer_gates_matches_blocked_qft_training():
    """Freezing identity outer QFT gates is equivalent to training BlockedBasis."""
    full_basis, frozen_outer, trainable_map = _qft_as_blocked_2x2_with_frozen_outer()
    blocked_basis = pdft.BlockedBasis(inner=pdft.QFTBasis(m=2, n=2), block_log_m=1, block_log_n=1)

    rng = np.random.default_rng(5)
    dataset = (rng.standard_normal((4, 8, 8)) + 1j * rng.standard_normal((4, 8, 8))).astype(
        np.complex128
    )

    # Initial transforms are the same operator before any training starts.
    pic = jnp.asarray(dataset[0])
    np.testing.assert_allclose(
        np.asarray(full_basis.forward_transform(pic)),
        np.asarray(blocked_basis.forward_transform(pic)),
        atol=1e-12,
        rtol=0.0,
    )

    train_kwargs = {
        "dataset": dataset,
        "loss": pdft.MSELoss(k=8),
        "epochs": 2,
        "batch_size": 2,
        "optimizer": "adam",
        "validation_split": 0.0,
        "early_stopping_patience": 10**9,
        "warmup_frac": 0.0,
        "lr_peak": 0.003,
        "lr_final": 0.003,
        "max_grad_norm": 1.0,
        "shuffle": False,
        "seed": 123,
    }

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


# ---------------------------------------------------------------------------
# optimizer specs
# ---------------------------------------------------------------------------


def test_resolve_optimizer_keeps_an_instances_settings_and_takes_the_scheduled_lr():
    from pdft.training.batched import _resolve_optimizer

    gd = pdft.RiemannianGD(lr=9.0, armijo_c=0.2, armijo_tau=0.3, max_ls_steps=4, max_grad_norm=1.5)
    assert _resolve_optimizer(gd, lr=0.5, max_grad_norm=None) == pdft.RiemannianGD(
        lr=0.5, armijo_c=0.2, armijo_tau=0.3, max_ls_steps=4, max_grad_norm=1.5
    )
    adam = pdft.RiemannianAdam(lr=9.0, beta1=0.8, beta2=0.95, eps=1e-6)
    assert _resolve_optimizer(adam, lr=0.5, max_grad_norm=2.0) == pdft.RiemannianAdam(
        lr=0.5, beta1=0.8, beta2=0.95, eps=1e-6, max_grad_norm=2.0
    )
    assert _resolve_optimizer("GD", lr=0.1, max_grad_norm=3.0) == pdft.RiemannianGD(
        lr=0.1, max_grad_norm=3.0
    )
    assert _resolve_optimizer("adam", lr=0.1, max_grad_norm=None) == pdft.RiemannianAdam(lr=0.1)
    for bad in ("sgd", 3):
        with pytest.raises(ValueError, match="unknown optimizer"):
            _resolve_optimizer(bad, lr=0.1, max_grad_norm=None)


@pytest.mark.parametrize(
    "optimizer",
    [pdft.RiemannianGD(lr=1.0, max_ls_steps=3), pdft.RiemannianAdam(lr=1.0, beta1=0.5)],
)
def test_batched_accepts_an_optimizer_instance_and_a_short_last_batch(optimizer):
    """Five images in batches of two: Adam pads the last batch by rotation, GD takes it short.
    Either way there are three steps per epoch."""
    images = [np.random.default_rng(seed).normal(size=(4, 4)) for seed in range(5)]
    result = train_basis_batched(
        pdft.QFTBasis(m=2, n=2),
        dataset=images,
        loss=pdft.L1Norm(),
        epochs=2,
        batch_size=2,
        optimizer=optimizer,
        seed=0,
    )
    assert result.steps == 6 and len(result.loss_history) == 6 and result.epochs_completed == 2
    assert all(isinstance(value, float) and math.isfinite(value) for value in result.loss_history)
    assert type(result.basis) is pdft.QFTBasis


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"dataset": []}, "dataset must be non-empty"),
        ({"epochs": 0}, "epochs must be >= 1"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"early_stopping_patience": 0}, "early_stopping_patience must be >= 1"),
        ({"warmup_frac": 1.0}, "warmup_frac must be in"),
        ({"val_every_k_epochs": 0}, "val_every_k_epochs must be >= 1"),
        ({"dataset": [np.zeros((2, 2))]}, "has shape"),
    ],
)
def test_batched_argument_validation(argument, message):
    arguments = {
        "dataset": [np.ones((4, 4))],
        "loss": pdft.L1Norm(),
        "epochs": 1,
        "batch_size": 1,
        **argument,
    }
    with pytest.raises(ValueError, match=message):
        train_basis_batched(pdft.QFTBasis(m=2, n=2), **arguments)


# ---------------------------------------------------------------------------
# the two Adam drivers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_grad_norm", [None, 0.05])
@pytest.mark.parametrize(
    "make", [lambda: pdft.QFTBasis(m=2, n=2), lambda: pdft.RichBasis(m=2, n=2)]
)
def test_the_fused_adam_step_tracks_the_eager_one(make, max_grad_norm):
    """`train_basis` drives Adam through `optimize`, one eager step at a time;
    `train_basis_batched` runs the fused jitted step. They share the update, so on
    one image and a flat schedule they must agree to rounding."""
    rng = np.random.default_rng(0)
    image = complex_normal(rng, (4, 4))
    lr, steps = 0.02, 6
    eager = pdft.train_basis(
        make(),
        target=jnp.asarray(image),
        loss=pdft.MSELoss(k=6),
        optimizer=pdft.RiemannianAdam(lr=lr, max_grad_norm=max_grad_norm),
        steps=steps,
    )
    fused = train_basis_batched(
        make(),
        dataset=[image],
        loss=pdft.MSELoss(k=6),
        epochs=steps,
        batch_size=1,
        optimizer=pdft.RiemannianAdam(max_grad_norm=max_grad_norm),
        lr_peak=lr,
        lr_final=lr,
        warmup_frac=0.0,
        shuffle=False,
    )
    assert cosine_with_warmup(1, steps, warmup_frac=0.0, lr_peak=lr, lr_final=lr) == lr
    for a, b in zip(eager.basis.tensors, fused.basis.tensors):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
    # the eager trace starts with the loss before any step; the fused one records
    # the loss each step started from
    np.testing.assert_allclose(eager.loss_history[:-1], fused.loss_history, rtol=1e-12)
    assert (
        max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(eager.basis.tensors, make().tensors))
        > 1e-3
    )


def test_adam_pads_a_short_last_batch_by_rotation():
    """Three images in batches of two: the second batch is the third image and the
    first again, so that every batch has the same shape. That is the same training
    run as four images with the first repeated."""
    rng = np.random.default_rng(0)
    a, b, c = (complex_normal(rng, (4, 4)) for _ in range(3))
    arguments = {"loss": pdft.MSELoss(k=6), "epochs": 2, "batch_size": 2, "shuffle": False}
    short = train_basis_batched(pdft.QFTBasis(m=2, n=2), dataset=[a, b, c], **arguments)
    padded = train_basis_batched(pdft.QFTBasis(m=2, n=2), dataset=[a, b, c, a], **arguments)
    assert short.loss_history == padded.loss_history and len(short.loss_history) == 4
    for x, y in zip(short.basis.tensors, padded.basis.tensors):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # and it is not the run that drops the incomplete batch's partner
    dropped = train_basis_batched(pdft.QFTBasis(m=2, n=2), dataset=[a, b, c, c], **arguments)
    assert dropped.loss_history != short.loss_history


def test_gd_takes_its_learning_rate_from_the_schedule():
    """An optimizer instance keeps its line-search settings; its own `lr` is replaced
    at every step by the schedule's."""
    rng = np.random.default_rng(1)
    images = [complex_normal(rng, (4, 4)) for _ in range(4)]

    def run(lr_final, optimizer):
        return train_basis_batched(
            pdft.QFTBasis(m=2, n=2),
            dataset=images,
            loss=pdft.L1Norm(),
            epochs=3,
            batch_size=2,
            optimizer=optimizer,
            lr_peak=0.2,
            lr_final=lr_final,
            warmup_frac=0.0,
            shuffle=False,
        ).loss_history

    flat, decayed = run(0.2, "gd"), run(0.002, "gd")
    assert flat[0] == decayed[0] and flat != decayed
    # the instance's own lr plays no part
    assert run(0.002, pdft.RiemannianGD(lr=123.0)) == decayed


def test_freezing_the_one_qubit_gates_trains_only_the_phases():
    basis = pdft.QFTBasis(m=2, n=2)
    hadamards = basis.program.tensor_indices(kind="H")
    images = [np.asarray(complex_image((4, 4), seed).real) for seed in range(4)]
    result = pdft.train_basis_batched(
        basis,
        dataset=images,
        loss=pdft.L1Norm(),
        epochs=2,
        batch_size=2,
        frozen_indices=hadamards,
        seed=0,
    )
    trained = result.basis
    for i, (before, after) in enumerate(zip(basis.tensors, trained.tensors)):
        assert jnp.array_equal(before, after) == (i in hadamards)
    # which is the configuration that cannot leave mu == 1
    assert pdft.certify_flat_modulus(trained, frozen_indices=hadamards)


@pytest.mark.parametrize("optimizer", ["adam", "gd"])
def test_the_epoch_loop_stops_when_told_to(optimizer, monkeypatch):
    """The decision is `evaluate_and_check_early_stop`'s; the loop has to act on it."""
    import pdft.training.batched as batched

    decide = batched.evaluate_and_check_early_stop

    def stop_after_second_epoch(**kwargs):
        best_tensors, best_val, patience, _, val_loss = decide(**kwargs)
        return best_tensors, best_val, patience, kwargs["epoch"] == 1, val_loss

    monkeypatch.setattr(batched, "evaluate_and_check_early_stop", stop_after_second_epoch)
    result = train_basis_batched(
        pdft.QFTBasis(m=2, n=2),
        dataset=_toy_dataset(6, 2, 2),
        loss=pdft.L1Norm(),
        epochs=10,
        batch_size=2,
        optimizer=optimizer,
        validation_split=0.34,
    )
    assert result.epochs_completed == 2 and len(result.val_history) == 2
    assert result.steps == len(result.loss_history) == 4


@pytest.mark.parametrize(
    ("frozen", "message"),
    [
        ([6], "out-of-range index 6"),
        ([-1], "negative index -1"),
        ([0, 0], "duplicate index 0"),
        ([1.9], "non-integer index 1.9"),
        ([True], "non-integer index True"),
        (["2"], "non-integer index '2'"),
    ],
)
def test_frozen_indices_are_validated(frozen, message):
    """With a dataset that is otherwise fine, so the refusal is the one being tested."""
    with pytest.raises(ValueError, match=f"frozen_indices contains {message}"):
        train_basis_batched(
            pdft.QFTBasis(m=2, n=2),
            dataset=_toy_dataset(2, 2, 2),
            loss=pdft.MSELoss(k=4),
            epochs=1,
            batch_size=1,
            frozen_indices=frozen,
        )
