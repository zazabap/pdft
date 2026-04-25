"""Layer A: Julia metrics.json runs through Python report code unmodified."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from generate_report import main as generate_report_main

FIXTURE = Path(__file__).parent / "fixtures" / "julia_quickdraw_metrics.json"


@pytest.fixture
def julia_results_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "quickdraw_moderate_20260101-000000"
    (rd / "loss_history").mkdir(parents=True)
    shutil.copy(FIXTURE, rd / "metrics.json")
    return rd


def test_parse_julia_metrics(julia_results_dir: Path):
    """Julia metrics.json parses cleanly as JSON and has expected top-level keys."""
    data = json.loads((julia_results_dir / "metrics.json").read_text())
    # Julia's quickdraw run includes at least these baselines/bases.
    assert "fft" in data
    assert "dct" in data
    # At least one quantum basis present.
    assert any(k in data for k in ("qft", "entangled_qft", "tebd", "mera"))


def test_generate_report_on_julia_metrics(julia_results_dir: Path):
    """Run our report generator on Julia output. Must succeed without errors."""
    generate_report_main(julia_results_dir)
    # CSVs produced.
    assert (julia_results_dir / "timing_summary.csv").is_file()
    for m in ("mse", "psnr", "ssim"):
        assert (julia_results_dir / f"rate_distortion_{m}.csv").is_file()
        assert (julia_results_dir / "plots" / f"rate_distortion_{m}.pdf").is_file()


def test_keep_ratio_keys_julia_form(julia_results_dir: Path):
    """Julia's keep_ratio keys are '0.05','0.1','0.15','0.2' — match Python's str(float)."""
    data = json.loads((julia_results_dir / "metrics.json").read_text())
    for basis_name, entry in data.items():
        if "metrics" not in entry:
            continue
        keys = set(entry["metrics"].keys())
        # Either Julia's natural form OR the equivalent.
        assert keys.issubset({"0.05", "0.1", "0.15", "0.2"}), (
            f"{basis_name}: unexpected keep_ratio keys {keys}"
        )
        break  # one is enough
