"""Training pipelines.

Two trainers:
- train_basis        — single-target loop (Phase 1, upstream parity).
- train_basis_batched — multi-image / multi-epoch with cosine LR schedule,
                        validation + early stopping, JIT'd Adam fast path.
"""

from .batched import train_basis_batched
from .result import TrainingResult
from .schedules import cosine_with_warmup
from .single import train_basis

__all__ = [
    "TrainingResult",
    "cosine_with_warmup",
    "train_basis",
    "train_basis_batched",
]
