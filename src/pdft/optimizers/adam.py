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
    bc1 = 1.0 - beta1**iter_1_based
    bc2 = 1.0 - beta2**iter_1_based

    for manifold, _indices in state.manifold_groups.items():
        rg = rg_batches[manifold]
        m_state = adam_state["m"][manifold]
        v_state = adam_state["v"][manifold]

        m_state = beta1 * m_state + (1.0 - beta1) * rg
        v_state = beta2 * v_state + (1.0 - beta2) * jnp.real(jnp.conj(rg) * rg)

        direction = (m_state / bc1) / (jnp.sqrt(v_state / bc2) + opt.eps)

        old_batch = state.point_batches[manifold]
        ib = state.ibatch_cache.get(manifold)
        new_batch = manifold.retract(old_batch, -direction, opt.lr, I_batch=ib)
        m_state = manifold.transport(old_batch, new_batch, m_state)

        adam_state["m"][manifold] = m_state
        adam_state["v"][manifold] = v_state
        state.point_batches[manifold] = new_batch
