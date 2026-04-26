"""Direct test for the extracted evaluate_and_check_early_stop helper.

The helper itself is a simple state-update function with no JAX or basis
dependencies. Tests cover: do_eval scheduling, no-validation fallback,
patience increment, early-stopping trigger, best_val tracking.
"""

import jax.numpy as jnp
import pytest

from pdft.training.eval_loop import evaluate_and_check_early_stop


def _ts(*vals):
    return [jnp.asarray(v, dtype=jnp.complex128) for v in vals]


def test_no_validation_overwrites_best_each_epoch():
    current = _ts(1.0, 2.0)
    best = _ts(0.0, 0.0)
    new_best, _new_val, patience, stop, vl = evaluate_and_check_early_stop(
        epoch=0,
        epochs=5,
        val_every_k_epochs=1,
        val_imgs=[],
        val_loss_fn=lambda _: float("inf"),
        current_tensors=current,
        best_tensors=best,
        best_val=float("inf"),
        patience=0,
        early_stopping_patience=3,
    )
    assert all(jnp.allclose(a, b) for a, b in zip(new_best, current))
    assert not stop
    assert patience == 0
    assert jnp.isnan(vl)


def test_validation_improves_resets_patience():
    _, new_val, patience, stop, _vl = evaluate_and_check_early_stop(
        epoch=0,
        epochs=5,
        val_every_k_epochs=1,
        val_imgs=[1, 2],
        val_loss_fn=lambda _: 0.5,
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=2,
        early_stopping_patience=3,
    )
    assert new_val == pytest.approx(0.5)
    assert patience == 0
    assert not stop


def test_validation_no_improvement_increments_patience():
    _, new_val, patience, stop, _vl = evaluate_and_check_early_stop(
        epoch=1,
        epochs=5,
        val_every_k_epochs=1,
        val_imgs=[1, 2],
        val_loss_fn=lambda _: 1.5,
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=2,
        early_stopping_patience=3,
    )
    assert new_val == pytest.approx(1.0)
    assert patience == 3
    assert stop


def test_skipped_epoch_does_not_advance_patience():
    _, _, patience, stop, vl = evaluate_and_check_early_stop(
        epoch=0,
        epochs=5,
        val_every_k_epochs=2,
        val_imgs=[1, 2],
        val_loss_fn=lambda _: 99.0,
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=2,
        early_stopping_patience=3,
    )
    assert jnp.isnan(vl)
    assert patience == 2
    assert not stop


def test_final_epoch_always_evaluated():
    _, val, _, _, _ = evaluate_and_check_early_stop(
        epoch=4,
        epochs=5,
        val_every_k_epochs=10,
        val_imgs=[1],
        val_loss_fn=lambda _: 0.42,
        current_tensors=_ts(1.0),
        best_tensors=_ts(0.0),
        best_val=1.0,
        patience=0,
        early_stopping_patience=3,
    )
    assert val == pytest.approx(0.42)
