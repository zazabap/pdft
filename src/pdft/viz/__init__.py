"""Matplotlib helpers: training-loss plots and circuit schematics.

Optional `plot` extra. Imported by examples and benchmarks; not depended
on by core. `viz.loss` plots loss histories; `viz.circuit` draws the
einsum schematic for a basis.
"""

from .loss import (
    TrainingHistory,
    ema_smooth,
    plot_training_comparison,
    plot_training_loss,
    save_training_plots,
)
from .circuit import plot_circuit

__all__ = [
    "TrainingHistory",
    "ema_smooth",
    "plot_circuit",
    "plot_training_comparison",
    "plot_training_loss",
    "save_training_plots",
]
