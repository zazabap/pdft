"""Shared optimizer infrastructure: state setup + batched projection."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..manifolds import (
    AbstractRiemannianManifold,
    UnitaryManifold,
    _make_identity_batch,
    group_by_manifold,
    stack_tensors,
)

Array = jax.Array


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
