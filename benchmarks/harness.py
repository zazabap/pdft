"""Training wrappers for the benchmark harness.

`train_one_basis_batched` is the new entry point: trains one shared basis on
the full dataset using `pdft.train_basis_batched` (cosine LR, validation
split, early stopping). Mirrors the pipeline in
`ParametricDFT-Benchmarks.jl`. The legacy `train_one_basis` (single-target,
fresh-basis-per-image) is retained for backwards compatibility but is no
longer used by `run_dataset`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Importing pdft enables JAX x64 mode globally (per CLAUDE.md §5).
# It MUST come before any `import jax` / `import jax.numpy` so that
# x64 is set before JAX caches its dtype defaults.
import pdft
from pdft.io_json import _format_float_julia_like

import jax  # noqa: E402

from config import Preset  # noqa: E402


OPTIMIZER_REGISTRY: dict[str, Callable[..., Any]] = {
    "gd": lambda lr: pdft.RiemannianGD(lr=lr),
    "adam": lambda lr: pdft.RiemannianAdam(lr=lr),
}


@dataclass
class TrainResult:
    """Result of training a single shared basis on a dataset.

    `loss_history` is per-step training loss, `val_history` is per-epoch
    validation loss (empty when validation_split=0).
    """

    basis: Any
    loss_history: list[float]
    time: float  # wall-clock incl. JIT (Julia-compatible)
    warmup_s: float  # first-call JIT cost (single-batch run)
    val_history: list[float] = field(default_factory=list)
    epochs_completed: int = 0
    steps: int = 0


def _make_optimizer(name: str, lr: float):
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(f"unknown optimizer {name!r}; choices: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](lr=lr)


def train_one_basis_batched(
    basis_factory: Callable[[], Any],
    train_imgs: np.ndarray | list,
    preset: Preset,
    *,
    device: jax.Device,
) -> TrainResult:
    """Train one shared basis on the whole training set using
    `pdft.train_basis_batched`. Pins device via `jax.default_device(device)`
    and blocks on the resulting tensors so timing is honest on GPU.

    The first batch's wall-clock is reported as `warmup_s` (it dominates
    JIT-compile time); the total wall-clock is in `time`.
    """
    with jax.default_device(device):
        # Materialize images on the requested device. Doing this BEFORE
        # entering `with jax.default_device(device)` would put arrays on
        # whichever device JAX picked at process start (typically gpu 0),
        # making every batch a cross-device copy on `--gpu 1` runs.
        target_dataset = [
            jax.device_put(np.asarray(img).astype(np.complex128), device) for img in train_imgs
        ]

        basis = basis_factory()

        t_warm = time.perf_counter()
        # Single-batch warmup pass to trigger JIT — pdft itself caches the
        # compiled einsum, so subsequent batches are fast. We use
        # `train_basis_batched(epochs=1, batch_size=1)` for the first image.
        if target_dataset:
            warmup_res = pdft.train_basis_batched(
                basis,
                dataset=target_dataset[:1],
                loss=pdft.MSELoss(k=max(1, round(2 ** (basis.m + basis.n) * 0.1))),
                epochs=1,
                batch_size=1,
                optimizer=preset.optimizer,
                validation_split=0.0,
                early_stopping_patience=1,
                warmup_frac=preset.warmup_frac,
                lr_peak=preset.lr_peak,
                lr_final=preset.lr_final,
                max_grad_norm=preset.max_grad_norm,
                shuffle=False,
                seed=preset.seed,
            )
            for t in warmup_res.basis.tensors:
                jax.block_until_ready(t)
        warmup_s = time.perf_counter() - t_warm

        t0 = time.perf_counter()
        # Full run starts from a fresh factory-init basis (we do NOT keep the
        # warmup basis — its single-step drift would skew the schedule).
        basis = basis_factory()
        result = pdft.train_basis_batched(
            basis,
            dataset=target_dataset,
            loss=pdft.MSELoss(k=max(1, round(2 ** (basis.m + basis.n) * 0.1))),
            epochs=preset.epochs,
            batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=preset.early_stopping_patience,
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True,
            seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
        )
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    return TrainResult(
        basis=result.basis,
        loss_history=list(result.loss_history),
        time=elapsed,
        warmup_s=warmup_s,
        val_history=list(result.val_history),
        epochs_completed=result.epochs_completed,
        steps=result.steps,
    )


def train_one_basis(
    basis_factory: Callable[[], Any],
    target: np.ndarray,
    preset: Preset,
    *,
    device: jax.Device,
    is_first_image: bool = False,
) -> TrainResult:
    """Legacy single-target trainer (fresh basis per image).

    Retained for the optimizer-comparison demo at `examples/optimizer_benchmark.py`
    and any caller that still wants P-pairing behaviour. New code should use
    `train_one_basis_batched`.
    """
    optimizer = _make_optimizer(preset.optimizer, preset.lr_peak)

    with jax.default_device(device):
        target_jnp = jax.device_put(np.asarray(target).astype(np.complex128), device)
        basis = basis_factory()
        t0 = time.perf_counter()
        # Use n_train * epochs as effective step count for back-compat. The
        # legacy harness only ran one image at a time so `epochs` here meant
        # gradient steps on that image.
        result = pdft.train_basis(
            basis,
            target=target_jnp,
            loss=pdft.MSELoss(k=max(1, round(2 ** (basis.m + basis.n) * 0.1))),
            optimizer=optimizer,
            steps=preset.epochs,
            seed=preset.seed,
        )
        for t in result.basis.tensors:
            jax.block_until_ready(t)
        elapsed = time.perf_counter() - t0

    warmup = elapsed if is_first_image else 0.0
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
