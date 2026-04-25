"""Layer A: generate_report.py + plots/. Synthetic metrics.json input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate_report import main as generate_report_main


@pytest.fixture
def synthetic_results(tmp_path: Path) -> Path:
    rd = tmp_path / "quickdraw_smoke_20260101-000000"
    (rd / "loss_history").mkdir(parents=True)

    metrics = {
        "qft": {
            "metrics": {
                "0.05": {
                    "mean_mse": 0.02,
                    "std_mse": 0.005,
                    "mean_psnr": 17.0,
                    "std_psnr": 0.5,
                    "mean_ssim": 0.4,
                    "std_ssim": 0.03,
                    "nan_count": 0,
                },
                "0.1": {
                    "mean_mse": 0.01,
                    "std_mse": 0.003,
                    "mean_psnr": 20.0,
                    "std_psnr": 0.5,
                    "mean_ssim": 0.6,
                    "std_ssim": 0.03,
                    "nan_count": 0,
                },
            },
            "time": 12.3,
            "_pdft_py": {"warmup_s": 1.5, "device": "cpu"},
        },
        "tebd": {
            "metrics": {
                "0.05": {
                    "mean_mse": 0.018,
                    "std_mse": 0.004,
                    "mean_psnr": 17.5,
                    "std_psnr": 0.6,
                    "mean_ssim": 0.5,
                    "std_ssim": 0.04,
                    "nan_count": 0,
                },
                "0.1": {
                    "mean_mse": 0.009,
                    "std_mse": 0.002,
                    "mean_psnr": 20.5,
                    "std_psnr": 0.6,
                    "mean_ssim": 0.7,
                    "std_ssim": 0.04,
                    "nan_count": 0,
                },
            },
            "time": 15.6,
            "_pdft_py": {"warmup_s": 1.6, "device": "cpu"},
        },
        "mera": {"skipped": "incompatible_qubits"},
        "fft": {
            "metrics": {
                "0.05": {
                    "mean_mse": 0.03,
                    "std_mse": 0.006,
                    "mean_psnr": 15.0,
                    "std_psnr": 0.7,
                    "mean_ssim": 0.36,
                    "std_ssim": 0.03,
                    "nan_count": 0,
                },
                "0.1": {
                    "mean_mse": 0.019,
                    "std_mse": 0.004,
                    "mean_psnr": 17.3,
                    "std_psnr": 0.7,
                    "mean_ssim": 0.45,
                    "std_ssim": 0.03,
                    "nan_count": 0,
                },
            },
            "time": 0.07,
        },
    }
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=4))

    # Stub loss histories for trajectory plots.
    for basis in ("qft", "tebd"):
        (rd / "loss_history" / f"{basis}_loss.json").write_text(
            json.dumps([[1.0, 0.8, 0.6], [0.9, 0.7, 0.5]])
        )

    return rd


def test_generate_report_writes_csvs(synthetic_results: Path):
    generate_report_main(synthetic_results)
    assert (synthetic_results / "timing_summary.csv").is_file()
    assert (synthetic_results / "rate_distortion_mse.csv").is_file()
    assert (synthetic_results / "rate_distortion_psnr.csv").is_file()
    assert (synthetic_results / "rate_distortion_ssim.csv").is_file()


def test_generate_report_writes_pdfs(synthetic_results: Path):
    generate_report_main(synthetic_results)
    pdir = synthetic_results / "plots"
    for name in ("rate_distortion_mse", "rate_distortion_psnr", "rate_distortion_ssim"):
        f = pdir / f"{name}.pdf"
        assert f.is_file()
        # PDF magic bytes
        with open(f, "rb") as fh:
            assert fh.read(5) == b"%PDF-"
    # loss-trajectory PDF — name uses dataset slug from results_dir name.
    loss_pdf = pdir / "loss_trajectories_quickdraw.pdf"
    assert loss_pdf.is_file()
    with open(loss_pdf, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_csv_row_counts(synthetic_results: Path):
    generate_report_main(synthetic_results)
    # rate_distortion_mse.csv: 1 header + (n_bases × n_keep_ratios) rows.
    # 4 bases (qft, tebd, mera, fft); mera skipped → 2 rows of NaN; 2 keep_ratios.
    text = (synthetic_results / "rate_distortion_mse.csv").read_text().strip().splitlines()
    # Header + 4 bases × 2 keep_ratios = 9 lines.
    assert len(text) == 1 + 4 * 2


def test_idempotent(synthetic_results: Path):
    """Running twice produces the same files (no exceptions, overwrites in place)."""
    generate_report_main(synthetic_results)
    first = (synthetic_results / "timing_summary.csv").read_text()
    generate_report_main(synthetic_results)
    second = (synthetic_results / "timing_summary.csv").read_text()
    assert first == second
