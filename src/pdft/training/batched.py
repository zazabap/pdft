"""Multi-image, multi-epoch trainer with cosine LR + early stopping.

Mirror of `ParametricDFT.jl/src/training.jl::_train_basis_core`.

Adam takes a JIT'd fast path (training.adam_step._build_jit_adam_step)
with persistent moment buffers and padded batches (constant XLA shape).
GD falls through to the original Armijo line search via optimize() since
its loss-eval count is data-dependent and not JIT-friendly.

The eval+early-stopping bookkeeping is shared via training.eval_loop.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..bases.core import with_tensors
from ..loss import AbstractLoss, mean_loss
from ..optimizers import (
    RiemannianAdam,
    RiemannianGD,
    optimize,
)
from .adam_step import adam_stepper
from .eval_loop import evaluate_and_check_early_stop
from .result import TrainingResult
from .schedules import cosine_with_warmup

Array = jax.Array


def _resolve_optimizer(spec, lr: float, max_grad_norm: float | None):
    """Build a fresh optimizer instance with the given lr/max_grad_norm.

    Accepts either a string name (`"gd"`/`"adam"`) or a class
    (`RiemannianGD`/`RiemannianAdam`); the latter is reconstructed with new
    `lr` so the cosine schedule can vary the learning rate per step.
    """
    if isinstance(spec, str):
        name = spec.lower()
        if name in ("gd", "gradient_descent"):
            return RiemannianGD(lr=lr, max_grad_norm=max_grad_norm)
        if name in ("adam",):
            return RiemannianAdam(lr=lr, max_grad_norm=max_grad_norm)
        raise ValueError(f"unknown optimizer {spec!r}; choices: 'gd', 'adam'")
    if isinstance(spec, (RiemannianGD, RiemannianAdam)):
        clip = max_grad_norm if max_grad_norm is not None else spec.max_grad_norm
        return dataclasses.replace(spec, lr=lr, max_grad_norm=clip)
    raise ValueError(f"unknown optimizer spec: {spec!r}")


def _validate_batched_args(
    dataset: Sequence,
    epochs: int,
    batch_size: int,
    validation_split: float,
    early_stopping_patience: int,
    warmup_frac: float,
):
    if len(dataset) == 0:
        raise ValueError("dataset must be non-empty")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not (0.0 <= validation_split < 1.0):
        raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")
    if early_stopping_patience < 1:
        raise ValueError(f"early_stopping_patience must be >= 1, got {early_stopping_patience}")
    if not (0.0 <= warmup_frac < 1.0):
        raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}")


def _validate_frozen_indices(frozen_indices: list[int] | None, n_tensors: int) -> frozenset:
    """Validate and normalise ``frozen_indices``.

    Returns a ``frozenset[int]`` of validated frozen indices (empty set means
    no freezing).  Raises ``ValueError`` on any violation.
    """
    if frozen_indices is None or len(frozen_indices) == 0:
        return frozenset()
    seen: set[int] = set()
    for raw_i in frozen_indices:
        if isinstance(raw_i, bool):
            raise ValueError(
                f"frozen_indices contains non-integer index {raw_i!r}; "
                "all indices must be integers."
            )
        try:
            i = operator.index(raw_i)
        except TypeError as exc:
            raise ValueError(
                f"frozen_indices contains non-integer index {raw_i!r}; "
                "all indices must be integers."
            ) from exc
        if i < 0:
            raise ValueError(
                f"frozen_indices contains negative index {i}; "
                f"all indices must be in [0, {n_tensors - 1}]."
            )
        if i >= n_tensors:
            raise ValueError(
                f"frozen_indices contains out-of-range index {i}; "
                f"basis has {n_tensors} tensors (valid range 0..{n_tensors - 1})."
            )
        if i in seen:
            raise ValueError(
                f"frozen_indices contains duplicate index {i}; each index must appear at most once."
            )
        seen.add(i)
    return frozenset(seen)


def train_basis_batched(
    basis,
    *,
    dataset: Sequence,
    loss: AbstractLoss,
    epochs: int,
    batch_size: int,
    optimizer="adam",
    validation_split: float = 0.0,
    early_stopping_patience: int = 5,
    warmup_frac: float = 0.05,
    lr_peak: float = 0.01,
    lr_final: float = 0.001,
    max_grad_norm: float | None = None,
    shuffle: bool = True,
    seed: int = 0,
    val_every_k_epochs: int = 1,
    frozen_indices: list[int] | None = None,
) -> TrainingResult:
    """Multi-image, multi-epoch trainer with cosine LR schedule.

    Mirror of `ParametricDFT.jl/src/training.jl::_train_basis_core` (main).
    `optimizer` is `"adam"`, `"gd"`, or an optimizer instance whose settings
    are kept; the learning rate always comes from the cosine schedule
    (`lr_peak`, `lr_final`, `warmup_frac`).

    Parameters
    ----------
    frozen_indices : list[int] | None, optional
        List of integer indices into ``basis.tensors``.  Tensors at these
        indices are NOT updated during training — they stay at their initial
        values throughout.  Each step computes gradients on all tensors
        normally; the update is then suppressed for frozen indices BEFORE any
        optimizer state is mutated (so Adam's moment buffers for frozen indices
        remain zero).  Useful for experiments that train only a subset of a
        circuit's gates.

        Validation: all indices must satisfy ``0 <= i < len(basis.tensors)``,
        no duplicates are allowed, and an empty list is treated as ``None``
        (no-op).  ``ValueError`` is raised on mis-specification.
    """
    frozen_set = _validate_frozen_indices(frozen_indices, len(list(basis.tensors)))
    _validate_batched_args(
        dataset, epochs, batch_size, validation_split, early_stopping_patience, warmup_frac
    )
    if val_every_k_epochs < 1:
        raise ValueError(f"val_every_k_epochs must be >= 1, got {val_every_k_epochs}")

    expected_size = basis.image_size
    images = []
    for i, img in enumerate(dataset):
        arr = jnp.asarray(np.asarray(img), dtype=jnp.complex128)
        if arr.shape != expected_size:
            raise ValueError(f"dataset[{i}] has shape {arr.shape}, expected {expected_size}")
        images.append(arr)

    rng = np.random.default_rng(seed)
    n_images = len(images)
    indices = rng.permutation(n_images) if shuffle else np.arange(n_images)
    n_validation = int(np.clip(round(n_images * validation_split), 0, n_images - 1))
    val_idx = indices[:n_validation].tolist()
    train_idx = indices[n_validation:].tolist()

    train_imgs = [images[i] for i in train_idx]
    val_imgs = [images[i] for i in val_idx]

    batch_size = min(batch_size, max(1, len(train_imgs)))
    n_batches = math.ceil(len(train_imgs) / batch_size)
    total_steps = max(1, epochs * n_batches)

    _mean_loss = mean_loss(basis, loss)

    _val_stacked = jnp.stack(val_imgs, axis=0) if val_imgs else None
    _val_eval = jax.jit(_mean_loss) if val_imgs else None

    def _val_loss(tensors: list[Array]) -> float:
        if _val_stacked is None:
            return float("inf")
        return float(_val_eval(tensors, _val_stacked))

    current_tensors = [jnp.asarray(t) for t in basis.tensors]
    best_tensors = [jnp.asarray(t) for t in current_tensors]
    best_val = float("inf")
    patience = 0

    loss_history: list[float] = []
    val_history: list[float] = []
    global_step = 0
    epochs_completed = 0

    # The two optimisers share the epoch loop below. Each supplies how an
    # epoch's images are cut into batches and what one step does. The
    # learning rate of `spec` is a placeholder: the schedule sets it per step.
    spec = _resolve_optimizer(optimizer, lr=lr_peak, max_grad_norm=max_grad_norm)
    if isinstance(spec, RiemannianAdam):
        adam_step = adam_stepper(
            basis,
            loss,
            beta1=spec.beta1,
            beta2=spec.beta2,
            eps=spec.eps,
            max_grad_norm=spec.max_grad_norm,
            frozen_set=frozen_set,
        )
        pad_count = n_batches * batch_size - len(train_imgs)

        def _batches(imgs: list[Array]) -> list[list[Array]]:
            # Pad by rotation so every batch is exactly `batch_size`: the
            # jitted step then sees one shape and compiles once.
            padded = imgs + imgs[:pad_count]
            return [padded[b * batch_size : (b + 1) * batch_size] for b in range(n_batches)]

        def _step(tensors: list[Array], batch_imgs: list[Array], lr_t: float, step: int):
            return adam_step(tensors, jnp.stack(batch_imgs, axis=0), lr_t, step)

    else:
        # GD path (Armijo line search). The last batch may be short: nothing
        # is jitted on the batch shape here.
        def _batches(imgs: list[Array]) -> list[list[Array]]:
            return [imgs[start : start + batch_size] for start in range(0, len(imgs), batch_size)]

        def _step(tensors: list[Array], batch_imgs: list[Array], lr_t: float, step: int):
            stacked = jnp.stack(batch_imgs, axis=0)

            def batch_loss_fn(ts: list[Array]) -> Array:
                return _mean_loss(ts, stacked)

            tensors, step_trace = optimize(
                dataclasses.replace(spec, lr=lr_t),
                tensors,
                batch_loss_fn,
                jax.grad(batch_loss_fn, argnums=0),
                max_iter=1,
                tol=0.0,
                record_loss=True,
                frozen_indices=frozen_set,
            )
            return tensors, step_trace[-1] if len(step_trace) >= 2 else step_trace[0]

    t0 = time.perf_counter()
    for epoch in range(epochs):
        if shuffle and epoch > 0:
            order = rng.permutation(len(train_imgs))
            train_imgs = [train_imgs[i] for i in order]

        # The Adam step returns its loss as a device array; converting after
        # the epoch keeps the steps from waiting on each other's results.
        epoch_losses: list = []
        for batch_imgs in _batches(train_imgs):
            global_step += 1
            lr_t = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_frac=warmup_frac,
                lr_peak=lr_peak,
                lr_final=lr_final,
            )
            current_tensors, loss_val = _step(current_tensors, batch_imgs, lr_t, global_step)
            epoch_losses.append(loss_val)
        loss_history.extend(float(L) for L in epoch_losses)

        epochs_completed = epoch + 1
        best_tensors, best_val, patience, stop, val_loss = evaluate_and_check_early_stop(
            epoch=epoch,
            epochs=epochs,
            val_every_k_epochs=val_every_k_epochs,
            val_imgs=val_imgs,
            val_loss_fn=_val_loss,
            current_tensors=current_tensors,
            best_tensors=best_tensors,
            best_val=best_val,
            patience=patience,
            early_stopping_patience=early_stopping_patience,
        )
        val_history.append(val_loss)
        if stop:
            break

    elapsed = time.perf_counter() - t0

    return TrainingResult(
        basis=with_tensors(basis, best_tensors),
        loss_history=loss_history,
        seed=seed,
        steps=global_step,
        wall_time_s=elapsed,
        val_history=val_history,
        epochs_completed=epochs_completed,
    )
