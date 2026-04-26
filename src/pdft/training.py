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
from .manifolds import (
    UnitaryManifold,
    _make_identity_batch,
    group_by_manifold,
    stack_tensors,
)
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


def _build_jit_adam_step(
    basis,
    loss: AbstractLoss,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    max_grad_norm: float | None,
):
    """Build a single JIT'd Adam step for `train_basis_batched`'s fast path.

    Returns ``step_fn(tensors_list, m_list, v_list, batch, lr_arr, iter_arr)
    -> (new_tensors_list, new_m_list, new_v_list, loss_value)``.

    The function fuses forward+backward+project+retract+transport+adam-update
    into one compiled XLA program. ``lr`` and ``iter_arr`` are TRACED inputs
    so the cosine schedule does not trigger XLA recompiles. All other Adam
    hyperparameters are compile-time constants.

    The Adam moment buffers (``m``, ``v``) are passed in/out so the caller
    persists them across steps — this is the corrected behaviour matching
    Julia's ``ParametricDFT.jl``: moments accumulate across the whole training
    run rather than being re-zeroed every batch.
    """
    m_qb, n_qb = basis.m, basis.n
    code = basis.code
    inv_code = basis.inv_code

    def per_image_loss(tensors, img):
        return loss_function(tensors, m_qb, n_qb, code, img, loss, inverse_code=inv_code)

    batched_loss = jax.vmap(per_image_loss, in_axes=(None, 0))

    def stacked_loss(tensors, batch):
        return jnp.mean(batched_loss(tensors, batch))

    val_grad_fn = jax.value_and_grad(stacked_loss, argnums=0)

    # Pre-classify manifolds once — static across the whole training run.
    template_tensors = list(basis.tensors)
    groups = group_by_manifold(template_tensors)
    manifold_list = list(groups.keys())
    indices_list = [tuple(groups[mfd]) for mfd in manifold_list]

    # Pre-compute identity batches for unitary manifolds (closure constants
    # — avoids rebuilding them on every step inside `retract`).
    ibs = []
    for manifold, idxs in zip(manifold_list, indices_list):
        if isinstance(manifold, UnitaryManifold):
            d = template_tensors[idxs[0]].shape[0]
            ibs.append(_make_identity_batch(template_tensors[idxs[0]].dtype, d, len(idxs)))
        else:
            ibs.append(None)

    @jax.jit
    def step_fn(tensors_list, m_list, v_list, batch, lr, iter_1based):
        # Forward + backward; loss comes "for free" alongside grads.
        loss_val, raw_grads = val_grad_fn(tensors_list, batch)
        # Wirtinger conjugation: JAX returns ∂f/∂z̄, Julia Zygote returns ∂f/∂z.
        # See CLAUDE.md §1 — must stay or trajectories drift.
        grads = [jnp.conj(g) for g in raw_grads]

        # Per-manifold project (fused into the JIT'd graph).
        pb_list = []
        rg_list = []
        for manifold, idxs in zip(manifold_list, indices_list):
            pb = jnp.stack([tensors_list[i] for i in idxs], axis=-1)
            gb = jnp.stack([grads[i] for i in idxs], axis=-1)
            rg = manifold.project(pb, gb)
            pb_list.append(pb)
            rg_list.append(rg)

        # Optional global gradient clipping (compile-time branch).
        if max_grad_norm is not None:
            grad_norm_sq = jnp.zeros((), dtype=jnp.float64)
            for rg in rg_list:
                grad_norm_sq = grad_norm_sq + jnp.real(jnp.sum(jnp.conj(rg) * rg))
            grad_norm = jnp.sqrt(grad_norm_sq)
            clip = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-30))
            rg_list = [rg * clip for rg in rg_list]

        # Bias correction with TRACED iter — no recompile when iter changes.
        bc1 = 1.0 - beta1**iter_1based
        bc2 = 1.0 - beta2**iter_1based

        new_tensors = list(tensors_list)
        new_m_list = []
        new_v_list = []
        for k, (manifold, idxs, ib) in enumerate(zip(manifold_list, indices_list, ibs)):
            rg = rg_list[k]
            pb = pb_list[k]
            m_old = m_list[k]
            v_old = v_list[k]

            new_m = beta1 * m_old + (1.0 - beta1) * rg
            new_v = beta2 * v_old + (1.0 - beta2) * jnp.real(jnp.conj(rg) * rg)

            direction = (new_m / bc1) / (jnp.sqrt(new_v / bc2) + eps)
            new_pb = manifold.retract(pb, -direction, lr, I_batch=ib)
            new_m = manifold.transport(pb, new_pb, new_m)

            new_m_list.append(new_m)
            new_v_list.append(new_v)
            for k2, idx in enumerate(idxs):
                new_tensors[idx] = new_pb[:, :, k2]

        return new_tensors, new_m_list, new_v_list, loss_val

    return step_fn


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
    val_every_k_epochs :
        Run validation eval (and check early-stopping) every K epochs. Default
        1 = match Julia (every epoch). Setting K=2 halves the validation cost
        on heavy configs (e.g. m=n=10 where val pass is ~60s per epoch).
        early_stopping_patience now counts in *evaluations*, not epochs:
        K=2, patience=5 means 10 epochs without improvement triggers stop.
        The final epoch is always evaluated so best_tensors is always fresh.
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
        """Single-image loss closure. Used as the body for the vmap."""
        return loss_function(tensors, m, n, code, img, loss, inverse_code=inv_code)

    # Vectorise over the leading batch axis. Mirrors Julia's
    # `make_batched_code(optcode, n_gates)` + `optimize_batched_code(...)`
    # pattern in ParametricDFT.jl/src/training.jl.
    _batched_loss = jax.vmap(_per_image_loss, in_axes=(None, 0))

    # Validation: forward only (no optimiser step). JIT'd so per-epoch eval
    # avoids the unfused-Python overhead of plain vmap calls.
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

    # Branch on optimizer type. Adam takes the JIT'd fast path with persistent
    # moment buffers and padded batches (constant shape → single XLA compile).
    # GD falls through to the original Armijo-line-search path because its
    # per-step loss-evaluation count is data-dependent and not JIT-friendly.
    is_adam = (isinstance(optimizer, str) and optimizer.lower() == "adam") or isinstance(
        optimizer, RiemannianAdam
    )

    if is_adam:
        # Resolve Adam hyperparameters (compile-time constants for step_fn).
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

        # Pad train_imgs by rotation so every batch is exactly `batch_size`
        # → single XLA compile for the whole run. Without this, the last batch
        # of each epoch had a different shape and triggered a 10+s recompile
        # the first time it was seen.
        n_train_imgs = len(train_imgs)
        pad_count = n_batches * batch_size - n_train_imgs

        t0 = time.perf_counter()
        for epoch in range(epochs):
            if shuffle and epoch > 0:
                order = rng.permutation(n_train_imgs)
                train_imgs = [train_imgs[i] for i in order]

            padded_imgs = train_imgs + train_imgs[:pad_count] if pad_count > 0 else train_imgs

            # Accumulate per-step loss as JAX scalars; convert to Python floats
            # at end of epoch so the dispatcher can keep enqueuing steps.
            epoch_loss_arrs: list = []
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                batch_imgs = padded_imgs[start:end]
                stacked = jnp.stack(batch_imgs, axis=0)

                global_step += 1
                lr_t = _cosine_with_warmup(
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
            # Run validation only on the K-th epoch (and always on the last
            # epoch so best_tensors is fresh). Epochs without an eval still
            # advance the loop but skip the patience check.
            do_eval = val_imgs and ((epoch + 1) % val_every_k_epochs == 0 or epoch + 1 == epochs)
            val_loss = _val_loss(current_tensors) if do_eval else float("nan")
            val_history.append(val_loss)

            if val_imgs and do_eval:
                if val_loss < best_val:
                    best_val = val_loss
                    best_tensors = [jnp.asarray(t) for t in current_tensors]
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience and epoch > 0:
                        break
            elif not val_imgs:
                best_tensors = [jnp.asarray(t) for t in current_tensors]

        elapsed = time.perf_counter() - t0
    else:
        # GD path (Armijo line search). Closure rebuild + per-batch
        # `optimize(max_iter=1)` is preserved here — line search makes a
        # JIT'd fused step impractical for GD.
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
                loss_history.append(step_trace[-1] if len(step_trace) >= 2 else step_trace[0])
                global_step += 1

            epochs_completed = epoch + 1
            do_eval = val_imgs and ((epoch + 1) % val_every_k_epochs == 0 or epoch + 1 == epochs)
            val_loss = _val_loss(current_tensors) if do_eval else float("nan")
            val_history.append(val_loss)

            if val_imgs and do_eval:
                if val_loss < best_val:
                    best_val = val_loss
                    best_tensors = [jnp.asarray(t) for t in current_tensors]
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience and epoch > 0:
                        break
            elif not val_imgs:
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
