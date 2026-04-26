"""Multi-image, multi-epoch trainer with cosine LR + early stopping.

Mirror of `ParametricDFT.jl/src/training.jl::_train_basis_core`.

Adam takes a JIT'd fast path (training.adam_step._build_jit_adam_step)
with persistent moment buffers and padded batches (constant XLA shape).
GD falls through to the original Armijo line search via optimize() since
its loss-eval count is data-dependent and not JIT-friendly.

The eval+early-stopping bookkeeping is shared via training.eval_loop.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..loss import AbstractLoss, loss_function
from ..manifolds import group_by_manifold, stack_tensors
from ..optimizers import (
    RiemannianAdam,
    RiemannianGD,
    optimize,
)
from .adam_step import _build_jit_adam_step
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
) -> TrainingResult:
    """Multi-image, multi-epoch trainer with cosine LR schedule.

    Mirror of `ParametricDFT.jl/src/training.jl::_train_basis_core` (main).
    See parameter docs in the original training.py docstring.
    """
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

    m, n = basis.m, basis.n
    code = basis.code
    inv_code = basis.inv_code

    def _per_image_loss(tensors: list[Array], img: Array) -> Array:
        return loss_function(tensors, m, n, code, img, loss, inverse_code=inv_code)

    _batched_loss = jax.vmap(_per_image_loss, in_axes=(None, 0))

    _val_stacked = jnp.stack(val_imgs, axis=0) if val_imgs else None
    _val_eval = jax.jit(lambda ts, batch: jnp.mean(_batched_loss(ts, batch))) if val_imgs else None

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

    is_adam = (isinstance(optimizer, str) and optimizer.lower() == "adam") or isinstance(
        optimizer, RiemannianAdam
    )

    if is_adam:
        if isinstance(optimizer, RiemannianAdam):
            beta1, beta2, eps = optimizer.beta1, optimizer.beta2, optimizer.eps
            mgn_eff = max_grad_norm if max_grad_norm is not None else optimizer.max_grad_norm
        else:
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            mgn_eff = max_grad_norm

        step_fn = _build_jit_adam_step(
            basis,
            loss,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            max_grad_norm=mgn_eff,
        )

        # Initialise Adam moment buffers ONCE — they persist across all steps,
        # matching Julia's design and fixing the silent correctness bug where
        # max_iter=1 in the old path was zeroing m/v on every batch.
        groups_init = group_by_manifold(list(basis.tensors))
        m_state: list = []
        v_state: list = []
        for manifold, idxs in groups_init.items():
            pb = stack_tensors(list(basis.tensors), list(idxs))
            m_state.append(jnp.zeros_like(pb))
            v_state.append(jnp.zeros(pb.shape, dtype=jnp.float64))

        # Pad train_imgs by rotation so every batch is exactly `batch_size`.
        n_train_imgs = len(train_imgs)
        pad_count = n_batches * batch_size - n_train_imgs

        t0 = time.perf_counter()
        for epoch in range(epochs):
            if shuffle and epoch > 0:
                order = rng.permutation(n_train_imgs)
                train_imgs = [train_imgs[i] for i in order]

            padded_imgs = train_imgs + train_imgs[:pad_count] if pad_count > 0 else train_imgs

            epoch_loss_arrs: list = []
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                batch_imgs = padded_imgs[start:end]
                stacked = jnp.stack(batch_imgs, axis=0)

                global_step += 1
                lr_t = cosine_with_warmup(
                    global_step,
                    total_steps,
                    warmup_frac=warmup_frac,
                    lr_peak=lr_peak,
                    lr_final=lr_final,
                )

                current_tensors, m_state, v_state, loss_val = step_fn(
                    current_tensors,
                    m_state,
                    v_state,
                    stacked,
                    jnp.asarray(lr_t),
                    jnp.asarray(global_step, dtype=jnp.int32),
                )
                epoch_loss_arrs.append(loss_val)

            loss_history.extend(float(L) for L in epoch_loss_arrs)

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
    else:
        # GD path (Armijo line search).
        def _make_batch_loss_fn(batch_imgs: list[Array]):
            stacked = jnp.stack(batch_imgs, axis=0)

            def loss_fn(tensors: list[Array]) -> Array:
                return jnp.mean(_batched_loss(tensors, stacked))

            return loss_fn

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

                lr_t = cosine_with_warmup(
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
                loss_history.append(step_trace[-1] if len(step_trace) >= 2 else step_trace[0])
                global_step += 1

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
