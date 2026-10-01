"""Training-loss visualization helpers (matplotlib).

Mirror of upstream src/visualization.jl (essentials only). Requires the
`plot` extra: `pip install pdft[plot]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._figure import require_matplotlib, save


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


def _finish(fig, ax, title: str, output_path: str | Path | None):
    """Label the loss axes, save the figure when asked to, and return it."""
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, output_path)
    return fig


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
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(len(history.losses)))
    ax.plot(x, history.losses, label=history.label, alpha=0.7)
    if smooth_alpha is not None:
        ax.plot(
            x,
            ema_smooth(history.losses, alpha=smooth_alpha),
            linewidth=2,
            label=f"{history.label} (EMA α={smooth_alpha})",
        )
    return _finish(fig, ax, title, output_path)


def plot_training_comparison(
    histories: list[TrainingHistory],
    *,
    output_path: str | Path | None = None,
    title: str = "Training comparison",
):
    """Overlay multiple loss trajectories on one axis."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for h in histories:
        ax.plot(range(len(h.losses)), h.losses, label=h.label, alpha=0.8)
    return _finish(fig, ax, title, output_path)


def save_training_plots(
    histories: list[TrainingHistory],
    output_dir: str | Path,
    *,
    filename_prefix: str = "training",
) -> list[Path]:
    """Write one PNG per history plus a combined comparison plot. Returns paths."""
    require_matplotlib()
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
