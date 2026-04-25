"""Training loops for parametric bases.

`train_basis` is the single-target loop (Phase 1 of the upstream port).
`train_basis_batched` is the multi-image, multi-epoch loop with cosine LR
schedule, mini-batches, validation split + early stopping, and gradient
clipping — a Python port of
`ParametricDFT.jl/src/training.jl::_train_basis_core` (main branch).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from .loss import AbstractLoss, loss_function
from .optimizers import (
    AbstractRiemannianOptimizer,
    RiemannianAdam,
    RiemannianGD,
    optimize,
)

Array = jax.Array


@dataclass
class TrainingResult:
    basis: Any  # any registered basis type
    loss_history: list[float]
    seed: int
    steps: int
    wall_time_s: float
    val_history: list[float] = field(default_factory=list)
    epochs_completed: int = 0


# ---------------------------------------------------------------------------
# Single-target training (unchanged Phase 1 API)
# ---------------------------------------------------------------------------


def train_basis(
    basis,
    *,
    target: Array,
    loss: AbstractLoss,
    optimizer: AbstractRiemannianOptimizer,
    steps: int,
    seed: int = 0,
    device: str = "cpu",
) -> TrainingResult:
    """Train `basis` to minimize `loss(basis.tensors, target)` over `steps`.

    Works for any basis registered as a JAX pytree whose leaves begin with
    the forward-circuit tensor list followed by the inverse-circuit tensor
    list (current convention for all four bases: QFTBasis, EntangledQFTBasis,
    TEBDBasis, MERABasis).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    m, n = basis.m, basis.n
    code = basis.code
    inv_code = basis.inv_code

    def loss_fn(tensors: list[Array]) -> Array:
        return loss_function(tensors, m, n, code, target, loss, inverse_code=inv_code)

    grad_fn = jax.grad(loss_fn, argnums=0)

    t0 = time.perf_counter()
    final_tensors, history = optimize(
        optimizer,
        list(basis.tensors),
        loss_fn,
        grad_fn,
        max_iter=steps,
        tol=0.0,
        record_loss=True,
    )
    elapsed = time.perf_counter() - t0

    # Per Julia's design (single tensor list), pytree leaves are just `tensors`.
    leaves, treedef = tree_util.tree_flatten(basis)
    n_fwd = len(basis.tensors)
    new_leaves = list(final_tensors) + list(leaves[n_fwd:])
    trained = tree_util.tree_unflatten(treedef, new_leaves)

    return TrainingResult(
        basis=trained,
        loss_history=history,
        seed=seed,
        steps=steps,
        wall_time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Batched training (Phase 2)
# ---------------------------------------------------------------------------


def _cosine_with_warmup(
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
    if isinstance(spec, RiemannianGD):
        return RiemannianGD(
            lr=lr,
            armijo_c=spec.armijo_c,
            armijo_tau=spec.armijo_tau,
            max_ls_steps=spec.max_ls_steps,
            max_grad_norm=max_grad_norm if max_grad_norm is not None else spec.max_grad_norm,
        )
    if isinstance(spec, RiemannianAdam):
        return RiemannianAdam(
            lr=lr,
            beta1=spec.beta1,
            beta2=spec.beta2,
            eps=spec.eps,
            max_grad_norm=max_grad_norm if max_grad_norm is not None else spec.max_grad_norm,
        )
    raise ValueError(f"unknown optimizer spec: {spec!r}")


def _validate_batched_args(
    dataset: Sequence,
    epochs: int,
    batch_size: int,
    validation_split: float,
    early_stopping_patience: int,
    warmup_frac: float,
):
    if not dataset:
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
) -> TrainingResult:
    """Multi-image, multi-epoch trainer with cosine LR schedule.

    Mirror of `ParametricDFT.jl/src/training.jl::_train_basis_core` (main).

    Parameters
    ----------
    basis :
        The starting basis. One shared basis is updated across the whole run.
    dataset :
        Sequence of images (each shape == basis.image_size, complex- or
        real-valued). The trainer mean-averages the loss across the batch.
    loss :
        Loss object (e.g. `pdft.L1Norm()`, `pdft.MSELoss(k)`).
    epochs :
        Number of full passes over the training split.
    batch_size :
        Mini-batch size. Each optimizer step uses one batch.
    optimizer :
        `"gd"`, `"adam"`, or a RiemannianGD / RiemannianAdam instance whose
        per-step `lr` is overwritten by the cosine schedule.
    validation_split :
        Fraction in [0, 1) of the dataset reserved for validation.
        `validation_split=0` disables validation and early stopping.
    early_stopping_patience :
        Number of consecutive epochs without val-loss improvement before
        training stops. Ignored when `validation_split == 0`.
    warmup_frac :
        Fraction of total steps used for linear LR warmup (Julia default 0.05).
    lr_peak, lr_final :
        Peak (post-warmup) and final learning rates.
    max_grad_norm :
        Optional gradient-clipping threshold (passed to the per-step optimizer).
    shuffle :
        Shuffle training images each epoch (default True; matches Julia).
    seed :
        Seeds train/val split and per-epoch shuffles. Use the same seed for
        reproducible runs.
    """
    _validate_batched_args(
        dataset, epochs, batch_size, validation_split, early_stopping_patience, warmup_frac
    )

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

    m, n = basis.m, basis.n
    code = basis.code
    inv_code = basis.inv_code

    def _per_image_loss(tensors: list[Array], img: Array) -> Array:
        """Single-image loss closure. Used as the body for the vmap."""
        return loss_function(tensors, m, n, code, img, loss, inverse_code=inv_code)

    # Vectorise over the leading batch axis. Mirrors Julia's
    # `make_batched_code(optcode, n_gates)` + `optimize_batched_code(...)`
    # pattern in ParametricDFT.jl/src/training.jl: the entire batch passes
    # through one fused circuit application instead of a Python loop. This is
    # what closes the per-step optimisation gap with Julia (the loss
    # mathematics is identical to the loop version, but the gradient
    # numerical accumulation is more stable and the kernel-launch count drops
    # from O(batch_size) to O(1)).
    _batched_loss = jax.vmap(_per_image_loss, in_axes=(None, 0))

    def _make_batch_loss_fn(batch_imgs: list[Array]):
        # Stack the batch along a new leading axis; the vmap spreads it.
        stacked = jnp.stack(batch_imgs, axis=0)

        def loss_fn(tensors: list[Array]) -> Array:
            return jnp.mean(_batched_loss(tensors, stacked))

        return loss_fn

    # Validation: also vectorised, but only forward (no optimiser step).
    if val_imgs:
        _val_stacked = jnp.stack(val_imgs, axis=0)
    else:
        _val_stacked = None

    def _val_loss(tensors: list[Array]) -> float:
        if _val_stacked is None:
            return float("inf")
        return float(jnp.mean(_batched_loss(tensors, _val_stacked)))

    current_tensors = [jnp.asarray(t) for t in basis.tensors]
    best_tensors = [jnp.asarray(t) for t in current_tensors]
    best_val = float("inf")
    patience = 0

    loss_history: list[float] = []
    val_history: list[float] = []
    global_step = 0
    epochs_completed = 0

    t0 = time.perf_counter()
    for epoch in range(epochs):
        if shuffle and epoch > 0:
            order = rng.permutation(len(train_imgs))
            train_imgs = [train_imgs[i] for i in order]

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(train_imgs))
            if start >= end:
                continue
            batch_imgs = train_imgs[start:end]

            batch_loss_fn = _make_batch_loss_fn(batch_imgs)
            batch_grad_fn = jax.grad(batch_loss_fn, argnums=0)

            lr_t = _cosine_with_warmup(
                global_step + 1,
                total_steps,
                warmup_frac=warmup_frac,
                lr_peak=lr_peak,
                lr_final=lr_final,
            )
            opt_t = _resolve_optimizer(optimizer, lr=lr_t, max_grad_norm=max_grad_norm)

            current_tensors, step_trace = optimize(
                opt_t,
                current_tensors,
                batch_loss_fn,
                batch_grad_fn,
                max_iter=1,
                tol=0.0,
                record_loss=True,
            )
            # `optimize` with record_loss returns [initial_loss, post_step_loss]
            # for max_iter=1; we want just the post-step entry.
            loss_history.append(step_trace[-1] if len(step_trace) >= 2 else step_trace[0])
            global_step += 1

        epochs_completed = epoch + 1
        val_loss = _val_loss(current_tensors) if val_imgs else float("nan")
        val_history.append(val_loss)

        if val_imgs:
            if val_loss < best_val:
                best_val = val_loss
                best_tensors = [jnp.asarray(t) for t in current_tensors]
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience and epoch > 0:
                    break
        else:
            best_tensors = [jnp.asarray(t) for t in current_tensors]

    elapsed = time.perf_counter() - t0

    # Per Julia's design (single tensor list), pytree leaves are just `tensors`.
    leaves, treedef = tree_util.tree_flatten(basis)
    n_fwd = len(basis.tensors)
    new_leaves = list(best_tensors) + list(leaves[n_fwd:])
    trained = tree_util.tree_unflatten(treedef, new_leaves)

    return TrainingResult(
        basis=trained,
        loss_history=loss_history,
        seed=seed,
        steps=global_step,
        wall_time_s=elapsed,
        val_history=val_history,
        epochs_completed=epochs_completed,
    )
