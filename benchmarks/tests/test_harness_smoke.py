"""Layer A: harness.py CPU smoke test. No GPU. <30s wall-clock."""

from __future__ import annotations

import json
import time

import jax
import numpy as np
import pytest

import pdft

from config import Preset
from harness import (
    OPTIMIZER_REGISTRY,
    TrainResult,
    dump_metrics_json,
    train_one_basis,
    train_one_basis_batched,
)


@pytest.fixture
def cpu_device():
    return jax.devices("cpu")[0]


def _smoke_preset(epochs: int = 2, n_train: int = 2, batch_size: int = 1) -> Preset:
    return Preset(
        name="smoke",
        epochs=epochs,
        n_train=n_train,
        n_test=n_train,
        optimizer="gd",
        batch_size=batch_size,
        warmup_frac=0.0,
        lr_peak=0.01,
        lr_final=0.01,
        max_grad_norm=None,
        validation_split=0.0,
        early_stopping_patience=2,
    )


def test_optimizer_registry():
    assert "gd" in OPTIMIZER_REGISTRY
    assert "adam" in OPTIMIZER_REGISTRY
    opt = OPTIMIZER_REGISTRY["gd"](lr=0.01)
    assert isinstance(opt, pdft.RiemannianGD)


def test_optimizer_unknown_raises():
    with pytest.raises(KeyError):
        OPTIMIZER_REGISTRY["sgd"](lr=0.01)  # type: ignore[index]


def test_train_one_basis_batched_smoke(cpu_device):
    """train_one_basis_batched runs end-to-end on a tiny dataset."""
    rng = np.random.default_rng(0)
    images = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(3)]
    preset = _smoke_preset(epochs=2, n_train=3, batch_size=1)

    t0 = time.perf_counter()
    result = train_one_basis_batched(
        lambda: pdft.QFTBasis(m=2, n=2),
        images,
        preset,
        device=cpu_device,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert isinstance(result, TrainResult)
    # 2 epochs × ceil(3 / 1) = 6 steps.
    assert len(result.loss_history) == 6
    assert result.time > 0
    assert result.warmup_s > 0  # warmup pass populated
    assert result.epochs_completed == 2
    assert result.steps == 6


def test_train_one_basis_batched_validation_split(cpu_device):
    rng = np.random.default_rng(0)
    images = [rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) for _ in range(5)]
    preset = Preset(
        name="smoke",
        epochs=2,
        n_train=5,
        n_test=5,
        optimizer="gd",
        batch_size=2,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=None,
        validation_split=0.2,
        early_stopping_patience=10,
    )
    result = train_one_basis_batched(
        lambda: pdft.QFTBasis(m=2, n=2),
        images,
        preset,
        device=cpu_device,
    )
    assert len(result.val_history) == 2  # one entry per completed epoch


def test_train_one_basis_legacy_smoke(cpu_device):
    """Legacy single-target trainer still works for back-compat."""
    rng = np.random.default_rng(0)
    target = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(np.complex128)
    preset = _smoke_preset(epochs=2, n_train=1, batch_size=1)

    result = train_one_basis(
        lambda: pdft.QFTBasis(m=2, n=2),
        target,
        preset,
        device=cpu_device,
        is_first_image=True,
    )
    assert isinstance(result, TrainResult)
    assert len(result.loss_history) == 2
    assert result.warmup_s > 0


def test_train_one_basis_subsequent_image_no_warmup(cpu_device):
    rng = np.random.default_rng(0)
    target = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))).astype(np.complex128)
    preset = _smoke_preset(epochs=2, n_train=1, batch_size=1)
    res = train_one_basis(
        lambda: pdft.QFTBasis(m=2, n=2),
        target,
        preset,
        device=cpu_device,
        is_first_image=False,
    )
    assert res.warmup_s == 0.0


def test_dump_metrics_json_roundtrip(tmp_path):
    metrics = {
        "qft": {
            "metrics": {
                "0.05": {
                    "mean_mse": 0.012,
                    "std_mse": 0.001,
                    "mean_psnr": 19.1,
                    "std_psnr": 0.7,
                    "mean_ssim": 0.5,
                    "std_ssim": 0.04,
                }
            },
            "time": 1.23,
        },
        "mera": {"skipped": "incompatible_qubits"},
    }
    out = tmp_path / "metrics.json"
    dump_metrics_json(metrics, out)
    parsed = json.loads(out.read_text())
    assert parsed["qft"]["time"] == 1.23
    assert parsed["mera"]["skipped"] == "incompatible_qubits"
    assert parsed["qft"]["metrics"]["0.05"]["mean_mse"] == 0.012


def test_dump_metrics_json_julia_float_format(tmp_path):
    metrics = {"x": {"time": 5e-7}}
    out = tmp_path / "metrics.json"
    dump_metrics_json(metrics, out)
    text = out.read_text()
    assert "5.0e-7" in text or "5e-7" in text
    assert "5e-07" not in text
