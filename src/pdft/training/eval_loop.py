"""Validation eval + early-stopping bookkeeping (shared by Adam and GD batched paths)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


def evaluate_and_check_early_stop(
    *,
    epoch: int,
    epochs: int,
    val_every_k_epochs: int,
    val_imgs: list,
    val_loss_fn: Callable[[list[Array]], float],
    current_tensors: list[Array],
    best_tensors: list[Array],
    best_val: float,
    patience: int,
    early_stopping_patience: int,
) -> tuple[list[Array], float, int, bool, float]:
    """Run validation (if scheduled), update best/patience state, decide whether to stop.

    Returns:
        (best_tensors, best_val, patience, stop, val_loss_recorded)

    Behavior identical to the bookkeeping previously inlined in both the
    Adam and GD paths of `train_basis_batched`. The only consolidation is
    that one helper now serves both branches; control flow is unchanged.

    - If validation isn't scheduled this epoch (per `val_every_k_epochs`),
      `val_loss_recorded` is NaN and patience is not advanced.
    - The final epoch is always evaluated so `best_tensors` stays fresh.
    - Without a validation set, `best_tensors` is overwritten on every
      epoch (no patience tracking).
    """
    do_eval = bool(val_imgs) and ((epoch + 1) % val_every_k_epochs == 0 or epoch + 1 == epochs)
    val_loss = val_loss_fn(current_tensors) if do_eval else float("nan")

    stop = False
    if val_imgs and do_eval:
        if val_loss < best_val:
            best_val = val_loss
            best_tensors = [jnp.asarray(t) for t in current_tensors]
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience and epoch > 0:
                stop = True
    elif not val_imgs:
        best_tensors = [jnp.asarray(t) for t in current_tensors]

    return best_tensors, best_val, patience, stop, val_loss
