"""Layer A: evaluation.py unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation import (
    aggregate_per_keep_ratio,
    compute_metrics,
    evaluate_baseline,
)


@pytest.fixture
def img_32():
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)


def test_compute_metrics_identical_image(img_32):
    m = compute_metrics(img_32, img_32)
    assert m["mse"] == 0.0
    assert m["psnr"] == float("inf") or m["psnr"] > 200.0
    assert m["ssim"] > 0.999  # near-1 (some skimage versions cap at <1.0)


def test_compute_metrics_clamps_to_unit_range(img_32):
    """Recovered values outside [0,1] are clamped before metrics."""
    bad = img_32 + 5.0  # all values > 1
    m = compute_metrics(img_32, bad)
    # After clamping, recovered is all 1.0; mse > 0
    assert m["mse"] > 0


def test_compute_metrics_handles_complex_recovered(img_32):
    """Recovered may be complex (FFT path); imag part dropped before clamp."""
    complex_recovered = img_32.astype(np.complex128) + 0.0j
    m = compute_metrics(img_32, complex_recovered)
    assert m["mse"] == pytest.approx(0.0, abs=1e-12)


def test_compute_metrics_psnr_matches_skimage(img_32):
    """PSNR formula 10*log10(1/MSE) matches skimage with data_range=1.0."""
    from skimage.metrics import peak_signal_noise_ratio

    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.1, size=img_32.shape)
    recovered = np.clip(img_32 + noise, 0.0, 1.0)
    m = compute_metrics(img_32, recovered)
    sk_psnr = peak_signal_noise_ratio(img_32, recovered, data_range=1.0)
    assert m["psnr"] == pytest.approx(sk_psnr, rel=1e-9, abs=1e-9)


def test_aggregate_handles_nan():
    """NaN in mse/psnr/ssim lists → nanmean / nanstd skip them."""
    metrics_list = [
        {"mse": 0.1, "psnr": 10.0, "ssim": 0.5},
        {"mse": float("nan"), "psnr": float("nan"), "ssim": float("nan")},
        {"mse": 0.2, "psnr": 7.0, "ssim": 0.4},
    ]
    agg = aggregate_per_keep_ratio(metrics_list)
    assert agg["mean_mse"] == pytest.approx(0.15)
    assert agg["std_mse"] == pytest.approx(np.nanstd([0.1, 0.2]))
    # 1 nan present
    assert agg["nan_count"] == 1


def test_evaluate_baseline_returns_schema(img_32):
    """evaluate_baseline returns (metrics_dict, elapsed_seconds) with the right shape."""
    images = np.stack([img_32, img_32 * 0.5, img_32 * 0.7], axis=0)

    def passthrough(img: np.ndarray, keep: float) -> np.ndarray:
        return img  # perfect reconstruction at any keep ratio

    metrics, elapsed = evaluate_baseline(passthrough, images, [0.05, 0.10])
    assert elapsed >= 0
    assert set(metrics.keys()) == {"0.05", "0.1"}  # str(0.1) == "0.1"
    for kr_str, vals in metrics.items():
        assert set(vals.keys()) >= {
            "mean_mse",
            "std_mse",
            "mean_psnr",
            "std_psnr",
            "mean_ssim",
            "std_ssim",
        }
        # passthrough → mse=0 → mean_mse=0
        assert vals["mean_mse"] == pytest.approx(0.0, abs=1e-12)


def test_evaluate_baseline_failure_records_nan(img_32):
    """A baseline that raises on one image → that image's metrics are nan; others succeed."""
    images = np.stack([img_32, img_32, img_32], axis=0)

    call_count = {"n": 0}

    def fail_on_second(img: np.ndarray, keep: float) -> np.ndarray:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return img

    metrics, _ = evaluate_baseline(fail_on_second, images, [0.10])
    vals = metrics["0.1"]
    # 2 successes (mse=0), 1 failure (nan) → nanmean = 0
    assert vals["mean_mse"] == pytest.approx(0.0, abs=1e-12)
