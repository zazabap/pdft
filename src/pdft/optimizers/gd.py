"""Riemannian gradient descent with Armijo backtracking line search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..manifolds import unstack_tensors
from .core import _OptimizationState

Array = jax.Array


@dataclass(frozen=True)
class RiemannianGD:
    """Riemannian gradient descent with Armijo backtracking line search.

    Mirror of upstream src/optimizers.jl:156-166. Defaults match upstream.
    """

    lr: float = 0.01
    armijo_c: float = 1e-4
    armijo_tau: float = 0.5
    max_ls_steps: int = 10
    max_grad_norm: float | None = None


def _armijo_step(
    opt: RiemannianGD,
    state: _OptimizationState,
    rg_batches: dict,
    loss_fn: Callable,
    grad_norm_sq: float,
    cached_loss: float,
) -> float:
    """Mirror of upstream src/optimizers.jl:231-275.

    Returns the accepted candidate loss, or NaN if line search exhausted.
    Mutates `state.point_batches` and `state.current_tensors` in place.
    """
    current_loss = float(loss_fn(state.current_tensors)) if jnp.isnan(cached_loss) else cached_loss
    alpha = opt.lr
    last_cands: dict = {}

    for _ in range(opt.max_ls_steps):
        for manifold, indices in state.manifold_groups.items():
            pb = state.point_batches[manifold]
            rg = rg_batches[manifold]
            ib = state.ibatch_cache.get(manifold)
            cand = manifold.retract(pb, -rg, alpha, I_batch=ib)
            last_cands[manifold] = cand
            unstack_tensors(cand, indices, into=state.current_tensors)

        candidate_loss = float(loss_fn(state.current_tensors))
        if candidate_loss <= current_loss - opt.armijo_c * alpha * grad_norm_sq:
            for manifold in state.manifold_groups:
                state.point_batches[manifold] = last_cands[manifold]
            return candidate_loss

        alpha *= opt.armijo_tau

    # Line search exhausted — use smallest-step candidate
    for manifold in state.manifold_groups:
        state.point_batches[manifold] = last_cands[manifold]
    return float("nan")
