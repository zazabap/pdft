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


@dataclass(frozen=True)
class RiemannianAdam:
    """Riemannian Adam optimizer (Becigneul & Ganea, 2019).

    Mirror of upstream src/optimizers.jl:173-183. Defaults match upstream.
    """

    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float | None = None


# Union of supported optimizer types for dispatch.
AbstractRiemannianOptimizer = RiemannianGD | RiemannianAdam


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


def _init_adam_state(state: _OptimizationState):
    """Mirror of upstream src/optimizers.jl:197-216.

    Returns a dict with per-manifold m (first moment, complex) and v
    (second moment, real) buffers, all initialized to zero.
    """
    m_buf: dict = {}
    v_buf: dict = {}
    for manifold, indices in state.manifold_groups.items():
        if not indices:
            continue
        pb = state.point_batches[manifold]
        m_buf[manifold] = jnp.zeros_like(pb)
        v_buf[manifold] = jnp.zeros(pb.shape, dtype=jnp.float64)
    return {"m": m_buf, "v": v_buf}


def _adam_step(
    opt: RiemannianAdam,
    state: _OptimizationState,
    rg_batches: dict,
    iter_1_based: int,
    adam_state: dict,
) -> None:
    """Mirror of upstream src/optimizers.jl:277-319.

    Update m, v, direction buffer; retract along -direction; transport
    m onto the new tangent space. Mutates `state.point_batches` and
    `adam_state` in place.
    """
    beta1, beta2 = opt.beta1, opt.beta2
    bc1 = 1.0 - beta1 ** iter_1_based
    bc2 = 1.0 - beta2 ** iter_1_based

    for manifold, _indices in state.manifold_groups.items():
        rg = rg_batches[manifold]
        m_state = adam_state["m"][manifold]
        v_state = adam_state["v"][manifold]

        # In-place (functional) moment update
        m_state = beta1 * m_state + (1.0 - beta1) * rg
        v_state = beta2 * v_state + (1.0 - beta2) * jnp.real(jnp.conj(rg) * rg)

        # Bias-corrected direction
        direction = (m_state / bc1) / (jnp.sqrt(v_state / bc2) + opt.eps)

        # Retract along -direction
        old_batch = state.point_batches[manifold]
        ib = state.ibatch_cache.get(manifold)
        new_batch = manifold.retract(old_batch, -direction, opt.lr, I_batch=ib)

        # Transport momentum (re-project onto new tangent space)
        m_state = manifold.transport(old_batch, new_batch, m_state)

        adam_state["m"][manifold] = m_state
        adam_state["v"][manifold] = v_state
        state.point_batches[manifold] = new_batch


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
    opt: "RiemannianGD | RiemannianAdam",
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
    adam_state = _init_adam_state(state) if isinstance(opt, RiemannianAdam) else None

    for iter_0 in range(max_iter):
        for manifold, indices in state.manifold_groups.items():
            unstack_tensors(
                state.point_batches[manifold], indices, into=state.current_tensors
            )

        raw_grads = grad_fn(state.current_tensors)
        # JAX and Julia's Zygote use opposite Wirtinger conventions for gradients
        # of real-valued functions of complex inputs: JAX returns ∂f/∂z̄ while
        # Julia returns ∂f/∂z. These are complex conjugates. To match Julia's
        # trajectory (and produce correct updates w.r.t. the real manifold
        # structure), we conjugate the raw gradient before projection.
        raw_grads = [jnp.conj(g) for g in raw_grads]
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

        if isinstance(opt, RiemannianGD):
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
                    trace.append(float(loss_fn(state.current_tensors)))
                else:
                    trace.append(float(cached_loss))
        elif isinstance(opt, RiemannianAdam):
            _adam_step(opt, state, rg_batches, iter_0 + 1, adam_state)
            if record_loss:
                # Adam does not evaluate loss; re-compute for the trace.
                for manifold, indices in state.manifold_groups.items():
                    unstack_tensors(
                        state.point_batches[manifold],
                        indices,
                        into=state.current_tensors,
                    )
                trace.append(float(loss_fn(state.current_tensors)))
        else:
            raise TypeError(f"unsupported optimizer type: {type(opt).__name__}")

    for manifold, indices in state.manifold_groups.items():
        unstack_tensors(
            state.point_batches[manifold], indices, into=state.current_tensors
        )

    return state.current_tensors, trace
