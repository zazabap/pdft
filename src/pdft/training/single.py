"""Single-target training loop: Phase 1 API, unchanged from upstream."""

from __future__ import annotations

import time

import jax

from ..bases.core import with_tensors
from ..loss import AbstractLoss, loss_function
from ..optimizers import AbstractRiemannianOptimizer, optimize
from .result import TrainingResult

Array = jax.Array


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
    its tensor list, the convention of every basis in the package.
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

    return TrainingResult(
        basis=with_tensors(basis, final_tensors),
        loss_history=history,
        seed=seed,
        steps=steps,
        wall_time_s=elapsed,
    )
