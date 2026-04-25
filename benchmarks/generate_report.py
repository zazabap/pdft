"""Aggregate metrics.json + loss histories into CSVs and PDF plots.

Idempotent: running on an existing results_dir overwrites outputs in place.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  -- adds benchmarks/ to sys.path

from plots.loss_trajectories import plot_loss_trajectories
from plots.rate_distortion import plot_rate_distortion

logger = logging.getLogger(__name__)


def _dataset_slug(results_dir: Path) -> str:
    """Extract dataset name from a results dir like 'quickdraw_smoke_20260101-000000'."""
    name = results_dir.name
    parts = name.split("_")
    # All preset names are single tokens; the timestamp is the last 1-2 tokens.
    # Drop trailing tokens that match a date-time pattern.
    while parts and parts[-1].replace("-", "").isdigit():
        parts.pop()
    if parts and parts[-1] in ("smoke", "moderate", "generalized"):
        parts.pop()
    return "_".join(parts) or "unknown"


def _write_timing_csv(metrics: dict, out_csv: Path) -> None:
    rows = [["basis", "time_s", "warmup_s"]]
    for name, data in sorted(metrics.items()):
        time_s = data.get("time", float("nan"))
        warmup_s = data.get("_pdft_py", {}).get("warmup_s", float("nan"))
        rows.append([name, _fmt(time_s), _fmt(warmup_s)])
    _write_csv(out_csv, rows)


def _write_rate_distortion_csv(metrics: dict, metric_name: str, out_csv: Path) -> None:
    """One row per (basis, keep_ratio). Skipped/failed bases produce one row per
    keep_ratio with NaN cells, using the union of keep_ratios seen across bases.
    """
    mean_key, std_key = f"mean_{metric_name}", f"std_{metric_name}"

    all_keep_ratios: set[str] = set()
    for data in metrics.values():
        if "metrics" in data:
            all_keep_ratios.update(data["metrics"].keys())
    if not all_keep_ratios:
        all_keep_ratios = {"0.05", "0.1", "0.15", "0.2"}
    keep_ratio_list = sorted(all_keep_ratios, key=float)

    rows = [["basis", "keep_ratio", "mean", "std"]]
    for basis_name in sorted(metrics.keys()):
        data = metrics[basis_name]
        for kr_str in keep_ratio_list:
            if "metrics" in data and kr_str in data["metrics"]:
                vals = data["metrics"][kr_str]
                rows.append([basis_name, kr_str, _fmt(vals[mean_key]), _fmt(vals[std_key])])
            else:
                rows.append([basis_name, kr_str, "NaN", "NaN"])
    _write_csv(out_csv, rows)


def _write_csv(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)


def _fmt(x) -> str:
    """Numeric -> string. NaN/None -> 'NaN'."""
    if x is None:
        return "NaN"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f != f:  # NaN
        return "NaN"
    return repr(f)


def main(results_dir: Path | str) -> None:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    dataset = _dataset_slug(results_dir)

    _write_timing_csv(metrics, results_dir / "timing_summary.csv")
    for m in ("mse", "psnr", "ssim"):
        _write_rate_distortion_csv(metrics, m, results_dir / f"rate_distortion_{m}.csv")
        plot_rate_distortion(metrics, m, results_dir / "plots" / f"rate_distortion_{m}.pdf")

    plot_loss_trajectories(
        results_dir / "loss_history",
        results_dir / "plots" / f"loss_trajectories_{dataset}.pdf",
        dataset,
    )
    logger.info("report generated under %s", results_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python benchmarks/generate_report.py <results_dir>")
        sys.exit(2)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main(sys.argv[1])
