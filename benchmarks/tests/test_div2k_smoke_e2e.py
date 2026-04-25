"""Layer B (opt-in): run_div2k_8q.py smoke end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


DIV2K_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


@pytest.mark.integration
def test_div2k_smoke_e2e(tmp_path: Path):
    if not DIV2K_ROOT.is_dir():
        pytest.skip(f"DIV2K not available at {DIV2K_ROOT}")
    from run_div2k_8q import main

    out_dir = tmp_path / "div2k_8q_smoke"
    rc = main(["smoke", "--allow-cpu", "--out", str(out_dir)])
    assert rc == 0
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # 4 bases + 4 baselines.
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
    # MERA either runs successfully OR fails due to memory constraints — both
    # are valid for a smoke test.  m+n=16 is a power of 2, so MERA is not
    # skipped; it will attempt to train and may OOM on CPU or low-memory GPUs.
    # The harness records OOM as {"failed": ...} rather than raising.
    assert "metrics" in metrics["mera"] or "failed" in metrics["mera"], (
        f"mera should attempt on DIV2K-8q; got {metrics['mera']}"
    )
