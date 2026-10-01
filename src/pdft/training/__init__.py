"""Training pipelines.

Two trainers:

- `train_basis` — `optimize` on one target image, which is what the Julia
  goldens harness runs (upstream has no one-image trainer of its own).
- `train_basis_batched` — upstream's `train_basis`: several images, epochs,
  a cosine LR schedule, validation + early stopping, JIT'd Adam fast path.
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
