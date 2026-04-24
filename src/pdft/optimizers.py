"""Riemannian optimizers.

Mirror of upstream src/optimizers.jl. Phase 1 implements RiemannianGD with
Armijo backtracking line search only; RiemannianAdam is Phase 2.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .manifolds import (
    AbstractRiemannianManifold,
    UnitaryManifold,
    _make_identity_batch,
    group_by_manifold,
    stack_tensors,
    unstack_tensors,
)

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


# ---------------------------------------------------------------------------
# Shared setup state
# ---------------------------------------------------------------------------


@dataclass
class _OptimizationState:
    manifold_groups: dict[AbstractRiemannianManifold, list[int]]
    point_batches: dict[AbstractRiemannianManifold, Array]
    ibatch_cache: dict[AbstractRiemannianManifold, Array]
    current_tensors: list[Array]


def _common_setup(tensors: list[Array]) -> _OptimizationState:
    """Mirror of upstream src/optimizers.jl:45-78."""
    groups = group_by_manifold(tensors)
    point_batches: dict[AbstractRiemannianManifold, Array] = {}
    ibatch_cache: dict[AbstractRiemannianManifold, Array] = {}
    for manifold, indices in groups.items():
        if not indices:
            continue
        pb = stack_tensors(tensors, indices)
        point_batches[manifold] = pb
        if isinstance(manifold, UnitaryManifold):
            d = pb.shape[0]
            n = len(indices)
            ibatch_cache[manifold] = _make_identity_batch(pb.dtype, d, n)
    return _OptimizationState(
        manifold_groups=groups,
        point_batches=point_batches,
        ibatch_cache=ibatch_cache,
        current_tensors=[jnp.asarray(t) for t in tensors],
    )


# ---------------------------------------------------------------------------
# Batched projection
# ---------------------------------------------------------------------------


def _batched_project(state: _OptimizationState, euclid_grads: list[Array]):
    """Mirror of upstream src/optimizers.jl:129-149."""
    rg_batches: dict[AbstractRiemannianManifold, Array] = {}
    grad_norm_sq = 0.0
    for manifold, indices in state.manifold_groups.items():
        pb = state.point_batches[manifold]
        gb = stack_tensors(euclid_grads, indices)
        rg = manifold.project(pb, gb)
        rg_batches[manifold] = rg
        grad_norm_sq = grad_norm_sq + float(jnp.real(jnp.sum(jnp.conj(rg) * rg)))
    return rg_batches, jnp.sqrt(grad_norm_sq)


# ---------------------------------------------------------------------------
# Armijo update step
# ---------------------------------------------------------------------------


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
    current_loss = (
        float(loss_fn(state.current_tensors)) if jnp.isnan(cached_loss) else cached_loss
    )
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize(
    opt: RiemannianGD,
    tensors: list[Array],
    loss_fn: Callable[[list[Array]], Array],
    grad_fn: Callable[[list[Array]], list[Array]],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    record_loss: bool = False,
) -> tuple[list[Array], list[float]]:
    """Mirror of upstream src/optimizers.jl:335-412.

    Returns (final_tensors, loss_history). `loss_history` is empty unless
    `record_loss=True`; then it starts with the initial loss and appends
    one entry per iteration.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if opt.lr <= 0:
        raise ValueError(f"opt.lr must be > 0, got {opt.lr}")

    state = _common_setup(tensors)
    trace: list[float] = []
    if record_loss:
        trace.append(float(loss_fn(state.current_tensors)))

    cached_loss = float("nan")

    for _ in range(max_iter):
        for manifold, indices in state.manifold_groups.items():
            unstack_tensors(
                state.point_batches[manifold], indices, into=state.current_tensors
            )

        raw_grads = grad_fn(state.current_tensors)
        for g in raw_grads:
            if not bool(jnp.all(jnp.isfinite(g))):
                warnings.warn("Non-finite gradient — optimizer stopping.", stacklevel=2)
                return state.current_tensors, trace

        rg_batches, grad_norm = _batched_project(state, raw_grads)
        grad_norm_sq = float(grad_norm) ** 2

        if opt.max_grad_norm is not None and float(grad_norm) > opt.max_grad_norm:
            clip = opt.max_grad_norm / float(grad_norm)
            rg_batches = {m: b * clip for m, b in rg_batches.items()}
            grad_norm_sq = opt.max_grad_norm ** 2

        if float(grad_norm) < tol:
            break

        cached_loss = _armijo_step(
            opt, state, rg_batches, loss_fn, grad_norm_sq, cached_loss
        )
        if record_loss:
            if jnp.isnan(cached_loss):
                for manifold, indices in state.manifold_groups.items():
                    unstack_tensors(
                        state.point_batches[manifold],
                        indices,
                        into=state.current_tensors,
                    )
                cached_loss = float(loss_fn(state.current_tensors))
            trace.append(float(cached_loss))

    for manifold, indices in state.manifold_groups.items():
        unstack_tensors(
            state.point_batches[manifold], indices, into=state.current_tensors
        )

    return state.current_tensors, trace
