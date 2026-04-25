"""Layer B (opt-in): run_all.sh fan-out. Skipped if <2 GPUs."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import jax
import pytest


@pytest.mark.integration
def test_2gpu_fanout_smoke(tmp_path: Path):
    if jax.default_backend() != "gpu":
        pytest.skip("no GPU backend")
    n_gpus = len(jax.devices("gpu"))
    if n_gpus < 2:
        pytest.skip(f"need 2 GPUs, have {n_gpus}")

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "run_all.sh"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    t0 = time.perf_counter()
    rc = subprocess.run(
        ["bash", str(script), "smoke"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    assert rc.returncode == 0, f"run_all.sh failed: stderr=\n{rc.stderr}"

    # Both result dirs should exist with metrics.json.
    qd = sorted((repo_root / "benchmarks" / "results").glob("quickdraw_smoke_*"))
    dv = sorted((repo_root / "benchmarks" / "results").glob("div2k_8q_smoke_*"))
    assert qd and (qd[-1] / "metrics.json").is_file()
    assert dv and (dv[-1] / "metrics.json").is_file()

    # Sanity: parallel run should complete faster than running sequentially —
    # but the absolute number depends heavily on GPU + dataset size, so we
    # only assert that elapsed is reasonable (<10 min).
    assert elapsed < 600
