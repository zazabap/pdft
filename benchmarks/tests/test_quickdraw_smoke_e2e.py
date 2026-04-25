"""Layer B (opt-in): run_quickdraw.py smoke end-to-end. CPU-allowed for CI runners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.integration
def test_quickdraw_smoke_e2e(tmp_path: Path):
    from run_quickdraw import main

    # Use a directory name that matches the expected slug pattern so that
    # _dataset_slug() in generate_report.py returns "quickdraw" and the loss
    # trajectory PDF is named "loss_trajectories_quickdraw.pdf".
    out_dir = tmp_path / "quickdraw_smoke"
    rc = main(["smoke", "--allow-cpu", "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "metrics.json").is_file()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # 4 quantum bases + 4 baselines = 8 keys.
    assert set(metrics.keys()) == {
        "qft",
        "entangled_qft",
        "tebd",
        "mera",
        "fft",
        "dct",
        "block_fft_8",
        "block_dct_8",
    }
    # MERA skipped on m=n=5.
    assert metrics["mera"].get("skipped") == "incompatible_qubits"
    # All PDFs.
    for name in (
        "rate_distortion_mse",
        "rate_distortion_psnr",
        "rate_distortion_ssim",
        "loss_trajectories_quickdraw",
    ):
        assert (out_dir / "plots" / f"{name}.pdf").is_file()
