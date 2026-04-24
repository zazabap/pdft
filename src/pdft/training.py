"""Single-target training loop for any registered basis.

Mirror of the inner optimization path in upstream src/training.jl. Phase 1
does not port batches, epochs, validation splits, LR schedules, or
checkpointing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
from jax import tree_util

from .loss import AbstractLoss, loss_function
from .optimizers import AbstractRiemannianOptimizer, optimize

Array = jax.Array


@dataclass
class TrainingResult:
    basis: Any  # any registered basis type
    loss_history: list[float]
    seed: int
    steps: int
    wall_time_s: float


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
        return loss_function(
            tensors, m, n, code, target, loss, inverse_code=inv_code
        )

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

    # Reconstruct a basis of the same concrete type with trained `tensors`.
    # We rely on the JAX pytree registration: leaves are (tensors..., inv_tensors...)
    # in that order, and aux data carries everything else (m, n, code, inv_code,
    # optional counts). Replacing the first len(tensors) leaves reconstructs
    # a same-typed basis.
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
