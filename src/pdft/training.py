"""Single-target training loop for AbstractSparseBasis instances.

Mirror of the inner optimization path in upstream src/training.jl. Phase 1
does not port batches, epochs, validation splits, LR schedules, or
checkpointing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax

from .basis import QFTBasis
from .loss import AbstractLoss, loss_function
from .optimizers import RiemannianGD, optimize

Array = jax.Array


@dataclass
class TrainingResult:
    basis: QFTBasis
    loss_history: list[float]
    seed: int
    steps: int
    wall_time_s: float


def train_basis(
    basis: QFTBasis,
    *,
    target: Array,
    loss: AbstractLoss,
    optimizer: RiemannianGD,
    steps: int,
    seed: int = 0,
    device: str = "cpu",
) -> TrainingResult:
    """Train `basis` to minimize `loss(basis.tensors, target)` over `steps`.

    `seed` is reserved for stochastic extensions; RiemannianGD itself is
    deterministic given identical inputs.
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

    trained = QFTBasis(
        m=m,
        n=n,
        tensors=final_tensors,
        inv_tensors=basis.inv_tensors,
        code=basis.code,
        inv_code=basis.inv_code,
    )
    return TrainingResult(
        basis=trained,
        loss_history=history,
        seed=seed,
        steps=steps,
        wall_time_s=elapsed,
    )
