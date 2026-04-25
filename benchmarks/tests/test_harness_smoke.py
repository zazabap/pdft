"""Layer A: harness.py CPU smoke test. No GPU. <10s wall-clock."""

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
)


@pytest.fixture
def cpu_device():
    """First CPU device. JAX always has at least one CPU device."""
    return jax.devices("cpu")[0]


def test_optimizer_registry():
    assert "gd" in OPTIMIZER_REGISTRY
    assert "adam" in OPTIMIZER_REGISTRY
    opt = OPTIMIZER_REGISTRY["gd"](lr=0.01)
    assert isinstance(opt, pdft.RiemannianGD)


def test_optimizer_unknown_raises():
    with pytest.raises(KeyError):
        OPTIMIZER_REGISTRY["sgd"](lr=0.01)  # type: ignore[index]


def test_train_one_basis_smoke(cpu_device):
    """2-step training on QFT m=2 n=2. Should complete <10s, return correct shape."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    target = target.astype(np.complex128)

    preset = Preset(name="smoke", epochs=2, n_train=1, n_test=1, optimizer="gd", lr=0.01)

    def factory():
        return pdft.QFTBasis(m=2, n=2)

    t0 = time.perf_counter()
    result = train_one_basis(factory, target, preset, device=cpu_device, is_first_image=True)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0

    assert isinstance(result, TrainResult)
    assert len(result.loss_history) == 2
    assert result.time > 0
    assert result.warmup_s > 0  # is_first_image=True → warmup populated
    # Loss should not increase over 2 GD steps on a small problem.
    assert result.loss_history[-1] <= result.loss_history[0] + 1e-10


def test_train_one_basis_subsequent_image_no_warmup(cpu_device):
    """is_first_image=False → warmup_s == 0."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    target = target.astype(np.complex128)
    preset = Preset(name="smoke", epochs=2, n_train=1, n_test=1, optimizer="gd", lr=0.01)
    res = train_one_basis(
        lambda: pdft.QFTBasis(m=2, n=2),
        target,
        preset,
        device=cpu_device,
        is_first_image=False,
    )
    assert res.warmup_s == 0.0


def test_dump_metrics_json_roundtrip(tmp_path):
    """dump_metrics_json produces parseable JSON with the expected keys."""
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
    """Very-small floats use Julia-style scientific notation (e.g. 5.0e-7 not 5e-07)."""
    metrics = {"x": {"time": 5e-7}}
    out = tmp_path / "metrics.json"
    dump_metrics_json(metrics, out)
    text = out.read_text()
    # Julia format: 5.0e-7 (mantissa has '.0', exponent has no leading zero, no '+').
    assert "5.0e-7" in text or "5e-7" in text  # tolerate either; spec leans Julia-style.
    assert "5e-07" not in text  # Python's default form must NOT appear.
