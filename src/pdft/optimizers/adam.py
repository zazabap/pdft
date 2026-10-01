"""Riemannian Adam (Becigneul & Ganea, 2019).

Note: this is the *general-purpose* Adam used by the optimize() dispatcher.
The batched training fast path (training/adam_step.py) uses a different
JIT-friendly representation (static lists indexed by k, not Python dicts
keyed by manifold) for XLA compilation; the duplication is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import _OptimizationState

Array = jax.Array


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


def _zero_moments(points: Array) -> tuple[Array, Array]:
    """Zero first (complex) and second (real) moment buffers for a batch of points."""
    return jnp.zeros_like(points), jnp.zeros(points.shape, dtype=jnp.float64)


def _adam_update(
    manifold,
    points: Array,
    rgrad: Array,
    m: Array,
    v: Array,
    *,
    lr,
    beta1: float,
    beta2: float,
    eps: float,
    bc1,
    bc2,
    I_batch: Array | None = None,
) -> tuple[Array, Array, Array]:
    """One Riemannian Adam update of a batch of points on one manifold.

    Mirror of upstream src/optimizers.jl:277-319: update the moments, step
    along the bias-corrected direction by retraction, transport the first
    moment to the new tangent space. Returns ``(new_points, new_m, new_v)``.

    Pure, with no Python control flow on values: `optimize` calls it eagerly
    and the batched trainer traces it inside its jitted step, so the two
    cannot drift apart. ``bc1`` and ``bc2`` are the bias corrections
    ``1 - beta**t`` and may be traced, like ``lr``.
    """
    m = beta1 * m + (1.0 - beta1) * rgrad
    v = beta2 * v + (1.0 - beta2) * jnp.real(jnp.conj(rgrad) * rgrad)
    direction = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    new_points = manifold.retract(points, -direction, lr, I_batch=I_batch)
    return new_points, manifold.transport(points, new_points, m), v


def _init_adam_state(state: _OptimizationState):
    """Mirror of upstream src/optimizers.jl:197-216.

    Returns a dict with per-manifold m (first moment, complex) and v
    (second moment, real) buffers, all initialized to zero.
    """
    moments = {manifold: _zero_moments(pb) for manifold, pb in state.point_batches.items()}
    return {
        "m": {manifold: m for manifold, (m, _) in moments.items()},
        "v": {manifold: v for manifold, (_, v) in moments.items()},
    }


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
    bc1 = 1.0 - opt.beta1**iter_1_based
    bc2 = 1.0 - opt.beta2**iter_1_based
    for manifold in state.manifold_groups:
        new_points, adam_state["m"][manifold], adam_state["v"][manifold] = _adam_update(
            manifold,
            state.point_batches[manifold],
            rg_batches[manifold],
            adam_state["m"][manifold],
            adam_state["v"][manifold],
            lr=opt.lr,
            beta1=opt.beta1,
            beta2=opt.beta2,
            eps=opt.eps,
            bc1=bc1,
            bc2=bc2,
            I_batch=state.ibatch_cache.get(manifold),
        )
        state.point_batches[manifold] = new_points
