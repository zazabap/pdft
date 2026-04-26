"""Learning-rate schedules for batched training."""

from __future__ import annotations

import math


def cosine_with_warmup(
    step: int,
    total_steps: int,
    *,
    warmup_frac: float = 0.05,
    lr_peak: float = 0.01,
    lr_final: float = 0.001,
) -> float:
    """Linear warmup followed by cosine decay.

    Mirror of `ParametricDFT.jl/src/training.jl::_cosine_with_warmup`.
    `step` is 0-indexed conceptually but Julia uses 1-indexed; we match
    Julia's behavior: the warmup ramp ends exactly at `step == warmup_steps`
    where `warmup_steps = max(1, round(warmup_frac * total_steps))`.
    """
    warmup_steps = max(1, round(warmup_frac * total_steps))
    if step <= warmup_steps:
        return lr_peak * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_final + 0.5 * (lr_peak - lr_final) * (1 + math.cos(math.pi * progress))
