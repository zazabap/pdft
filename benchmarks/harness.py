"""Single-basis training wrapper for the benchmark harness.

Wraps pdft.train_basis with timing and optional first-call JIT-warmup tracking.
Also provides a Julia-float-formatted JSON writer for metrics.json.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Importing pdft enables JAX x64 mode globally (per CLAUDE.md §5).
# It MUST come before any `import jax` / `import jax.numpy` so that
# x64 is set before JAX caches its dtype defaults.
import pdft
from pdft.io_json import _format_float_julia_like

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from config import Preset  # noqa: E402


OPTIMIZER_REGISTRY: dict[str, Callable[..., Any]] = {
    "gd": lambda lr: pdft.RiemannianGD(lr=lr),
    "adam": lambda lr: pdft.RiemannianAdam(lr=lr),
}


@dataclass
class TrainResult:
    basis: Any
    loss_history: list[float]
    time: float  # wall-clock incl. JIT (Julia-compatible)
    warmup_s: float  # wall-clock for first image (incl. JIT); 0 for subsequent images


def _make_optimizer(name: str, lr: float):
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(f"unknown optimizer {name!r}; choices: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](lr=lr)


def train_one_basis(
    basis_factory: Callable[[], Any],
    target: np.ndarray,
    preset: Preset,
    *,
    device: jax.Device,
    is_first_image: bool = False,
) -> TrainResult:
    """Train a fresh basis from `basis_factory()` on `target` for preset.epochs steps.

    Pins device via `jax.default_device(device)`. Calls `jax.block_until_ready`
    on the trained tensors before stopping the clock for honest GPU timing.
    """
    optimizer = _make_optimizer(preset.optimizer, preset.lr)
    target_jnp = jnp.asarray(target, dtype=jnp.complex128)

    with jax.default_device(device):
        basis = basis_factory()
        t0 = time.perf_counter()
        result = pdft.train_basis(
            basis,
            target=target_jnp,
            loss=pdft.L1Norm(),
            optimizer=optimizer,
            steps=preset.epochs,
            seed=preset.seed,
        )
        # Force completion of any in-flight async dispatch before stopping the clock.
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    warmup = elapsed if is_first_image else 0.0
    # pdft.train_basis returns initial_loss + one entry per step (steps+1 total).
    # TrainResult exposes only the per-step losses so len(loss_history) == preset.epochs.
    raw_history = list(result.loss_history)
    loss_history = raw_history[1:] if len(raw_history) > preset.epochs else raw_history
    return TrainResult(
        basis=result.basis,
        loss_history=loss_history,
        time=elapsed,
        warmup_s=warmup,
    )


def _julia_float_postprocess(json_text: str) -> str:
    """Rewrite Python-style scientific floats (5e-07) to Julia-style (5.0e-7).

    Python's `json` module uses `repr(float)` which yields forms like '5e-07'
    or '1.5e-07'. Julia's JSON3 uses Julia's `string(Float64)` which yields
    '5.0e-7' / '1.5e-7'. We match Julia's form in-place via regex.
    """
    pattern = re.compile(r"([-+]?\d+(?:\.\d+)?)e([-+]?\d+)")

    def fix(match: re.Match) -> str:
        mantissa = match.group(1)
        exponent = match.group(2)
        try:
            return _format_float_julia_like(float(f"{mantissa}e{exponent}"))
        except ValueError:
            return match.group(0)

    return pattern.sub(fix, json_text)


def dump_metrics_json(payload: dict, path: Path | str) -> None:
    """Write metrics.json with Julia-style float formatting in scientific notation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=4, allow_nan=True)
    text = _julia_float_postprocess(text)
    path.write_text(text)
