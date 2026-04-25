"""Rate-distortion plot: metric vs keep_ratio, one curve per basis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
matplotlib.rcParams["pdf.fonttype"] = 42  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

_METRIC_LABELS = {
    "mse": ("Mean Squared Error", "MSE (lower is better)"),
    "psnr": ("Peak Signal-to-Noise Ratio (dB)", "PSNR — higher is better"),
    "ssim": ("Structural Similarity", "SSIM — higher is better"),
}


def plot_rate_distortion(metrics: dict, metric_name: str, out_pdf: Path) -> None:
    """One panel per dataset (caller passes per-dataset metrics dict).

    `metric_name` in {"mse", "psnr", "ssim"}.
    """
    if metric_name not in _METRIC_LABELS:
        raise ValueError(f"unknown metric {metric_name!r}")
    ylabel, title_suffix = _METRIC_LABELS[metric_name]
    mean_key, std_key = f"mean_{metric_name}", f"std_{metric_name}"

    fig, ax = plt.subplots(figsize=(7, 5))
    for basis_name, data in sorted(metrics.items()):
        if "metrics" not in data:
            continue  # skipped or failed basis
        kr_strs = sorted(data["metrics"].keys(), key=float)
        xs = [float(k) for k in kr_strs]
        ys = [data["metrics"][k][mean_key] for k in kr_strs]
        errs = [data["metrics"][k][std_key] for k in kr_strs]
        ax.errorbar(xs, ys, yerr=errs, marker="o", label=basis_name, capsize=3)

    ax.set_xlabel("Keep ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title_suffix)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
