"""Training-loss visualization helpers (matplotlib).

Mirror of upstream src/visualization.jl (essentials only). Requires the
`plot` extra: `pip install pdft[plot]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            "matplotlib is required for pdft.viz. Install with: pip install pdft[plot]"
        ) from e


@dataclass
class TrainingHistory:
    """Thin wrapper around a loss trajectory for plotting."""
    losses: list[float]
    label: str = "training"


def ema_smooth(values, alpha: float = 0.1) -> list[float]:
    """Exponential moving average smoother. Returns a list of same length."""
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1 - alpha) * out[-1])
    return out


def plot_training_loss(
    history: TrainingHistory,
    *,
    output_path: str | Path | None = None,
    title: str = "Training loss",
    smooth_alpha: float | None = None,
):
    """Plot a single loss trajectory (optionally smoothed) and return the Figure.

    If `output_path` is given, saves the figure to that path and returns it.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(history.losses)))
    ax.plot(x, history.losses, label=history.label, alpha=0.7)
    if smooth_alpha is not None:
        ax.plot(x, ema_smooth(history.losses, alpha=smooth_alpha),
                linewidth=2, label=f"{history.label} (EMA α={smooth_alpha})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path is not None:
        fig.savefig(str(output_path), bbox_inches="tight", dpi=120)
    return fig


def plot_training_comparison(
    histories: list[TrainingHistory],
    *,
    output_path: str | Path | None = None,
    title: str = "Training comparison",
):
    """Overlay multiple loss trajectories on one axis."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in histories:
        ax.plot(range(len(h.losses)), h.losses, label=h.label, alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path is not None:
        fig.savefig(str(output_path), bbox_inches="tight", dpi=120)
    return fig


def save_training_plots(
    histories: list[TrainingHistory],
    output_dir: str | Path,
    *,
    filename_prefix: str = "training",
) -> list[Path]:
    """Write one PNG per history plus a combined comparison plot. Returns paths."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for h in histories:
        p = out_dir / f"{filename_prefix}_{h.label.replace(' ', '_')}.png"
        plot_training_loss(h, output_path=p)
        paths.append(p)
        plt.close()
    combined = out_dir / f"{filename_prefix}_comparison.png"
    plot_training_comparison(histories, output_path=combined)
    paths.append(combined)
    plt.close()
    return paths
