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
    unstack_tensors,
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


def _write_back(state: _OptimizationState) -> None:
    """Unstack every point batch into ``state.current_tensors``."""
    for manifold, indices in state.manifold_groups.items():
        unstack_tensors(state.point_batches[manifold], indices, into=state.current_tensors)


def _stack_grads(grads: list[Array], indices, frozen: frozenset[int] | None) -> Array:
    """Stack the gradients of one manifold group, with zeros for the frozen tensors.

    A zero gradient keeps a frozen tensor's Adam moments at zero, so nothing
    accumulates for it while the rest of its group trains.
    """
    if not frozen:
        return stack_tensors(grads, indices)
    return jnp.stack(
        [jnp.zeros_like(grads[i]) if i in frozen else grads[i] for i in indices], axis=-1
    )


def _batched_project(
    state: _OptimizationState,
    euclid_grads: list[Array],
    frozen_indices: frozenset[int] | None = None,
):
    """Mirror of upstream src/optimizers.jl:129-149."""
    rg_batches: dict[AbstractRiemannianManifold, Array] = {}
    grad_norm_sq = 0.0
    for manifold, indices in state.manifold_groups.items():
        pb = state.point_batches[manifold]
        rg = manifold.project(pb, _stack_grads(euclid_grads, indices, frozen_indices))
        rg_batches[manifold] = rg
        grad_norm_sq = grad_norm_sq + float(jnp.real(jnp.sum(jnp.conj(rg) * rg)))
    return rg_batches, jnp.sqrt(grad_norm_sq)
