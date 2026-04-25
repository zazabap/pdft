"""Loss-trajectory plot: one subplot panel per basis, n_train overlaid curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
matplotlib.rcParams["pdf.fonttype"] = 42  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_trajectories(loss_dir: Path, out_pdf: Path, dataset_name: str) -> None:
    """Reads *_loss.json files (list-of-lists) and produces one panel per basis."""
    files = sorted(loss_dir.glob("*_loss.json"))
    if not files:
        # No bases have loss histories (all skipped/failed) — write an empty PDF.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no loss histories", ha="center", va="center")
        ax.set_axis_off()
    else:
        n_panels = len(files)
        cols = min(2, n_panels)
        rows = (n_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
        axes_flat = axes.ravel()
        for ax, lf in zip(axes_flat, files):
            basis_name = lf.stem.replace("_loss", "")
            histories = json.loads(lf.read_text())
            for traj in histories:
                ax.plot(traj, alpha=0.6, linewidth=0.8)
            ax.set_title(f"{basis_name} — loss trajectories")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
        # Hide any unused subplots.
        for ax in axes_flat[len(files) :]:
            ax.set_axis_off()

    fig.suptitle(f"Loss trajectories — {dataset_name}", fontsize=12)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
