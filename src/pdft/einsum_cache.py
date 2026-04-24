"""In-memory cache for jit-compiled jnp.einsum closures.

Mirror of upstream src/einsum_cache.jl. Upstream persists TreeSA-optimized
paths to disk; XLA path optimization is cheap enough that an in-process
dict is sufficient for Phase 1.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

_CACHE: dict[tuple, Callable] = {}


def optimize_code_cached(subscripts: str, *shapes: tuple[int, ...]) -> Callable[..., jax.Array]:
    """Return a jit-compiled einsum closure bound to fixed subscripts and shapes.

    On first call for a given (subscripts, shapes) key, computes an optimal
    contraction path via jnp.einsum_path on dummy arrays of the supplied
    shapes, then returns a jit-compiled closure. Subsequent calls return the
    identical cached closure.

    Parameters
    ----------
    subscripts : str
        einsum subscripts string, e.g., "ijk,jlk->ilk".
    *shapes : tuple[int, ...]
        Shape of each operand, in order matching the subscripts.

    Returns
    -------
    Callable accepting the operands and returning the contraction result.
    """
    key = (subscripts, tuple(tuple(s) for s in shapes))
    if key in _CACHE:
        return _CACHE[key]

    dummies = [jnp.zeros(shape, dtype=jnp.complex128) for shape in shapes]
    # "greedy" scales polynomially in tensor count; "optimal" is exponential
    # and becomes impractical beyond ~10 tensors (a 3x3 QFT has 12).
    # For circuits of this size "greedy" matches "optimal" in practice.
    path, _info = jnp.einsum_path(subscripts, *dummies, optimize="greedy")

    @jax.jit
    def _contract(*operands):
        return jnp.einsum(subscripts, *operands, optimize=path)

    _CACHE[key] = _contract
    return _contract
