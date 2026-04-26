"""TrainingResult value object returned by both single and batched trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    basis: Any
    loss_history: list[float]
    seed: int
    steps: int
    wall_time_s: float
    val_history: list[float] = field(default_factory=list)
    epochs_completed: int = 0
