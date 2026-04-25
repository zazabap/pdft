"""Loss-trajectory plot: one subplot panel per basis, n_train overlaid curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
matplotlib.rcParams["pdf.fonttype"] = 42  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def _load_traces(payload):
    """Accept either the new dict shape (`{step_losses, val_losses, ...}`) or
    the legacy list-of-lists shape used by the per-image P-pair pipeline.
    Returns (step_traces, val_trace_or_None, step_xs).
    """
    if isinstance(payload, dict):
        step = payload.get("step_losses", [])
        val = payload.get("val_losses", []) or None
        return [step], val, list(range(1, len(step) + 1))
    # Legacy: list-of-lists, one trajectory per training image.
    return list(payload), None, None


def plot_loss_trajectories(loss_dir: Path, out_pdf: Path, dataset_name: str) -> None:
    """Reads *_loss.json files and produces one panel per basis."""
    files = sorted(loss_dir.glob("*_loss.json"))
    if not files:
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
            payload = json.loads(lf.read_text())
            traces, val_trace, _ = _load_traces(payload)
            for traj in traces:
                ax.plot(traj, alpha=0.7, linewidth=1.0, label="train")
            if val_trace:
                # Validation is per-epoch; place markers along the x-range so
                # the curve is visible even with one or two points.
                if traces and traces[0]:
                    n_steps = len(traces[0])
                    n_epochs = len(val_trace)
                    if n_epochs > 0:
                        xs = [int(round((i + 1) * (n_steps / n_epochs))) for i in range(n_epochs)]
                    else:
                        xs = []
                    ax.plot(xs, val_trace, "o-", color="tab:red", label="val", linewidth=1.2)
            ax.set_title(f"{basis_name} — loss trajectories")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            if val_trace:
                ax.legend(loc="best", fontsize=7)
        for ax in axes_flat[len(files) :]:
            ax.set_axis_off()

    fig.suptitle(f"Loss trajectories — {dataset_name}", fontsize=12)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
