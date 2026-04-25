"""Loss functions for sparse basis training.

Mirror of upstream src/loss.jl. Single-image path only; batched loss
dispatch is deferred to a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

Array = jax.Array


@runtime_checkable
class AbstractLoss(Protocol):
    """Marker protocol for loss types. No behavior; dispatch is functional."""

    ...


@dataclass(frozen=True)
class L1Norm:
    """L1 norm loss: minimizes `sum(|T(x)|)` to encourage sparsity."""

    pass


@dataclass(frozen=True)
class MSELoss:
    """MSE loss with top-k truncation: `||x - T^{-1}(truncate(T(x), k))||^2`.

    Parameter
    ---------
    k : int
        Number of coefficients to keep after top-k magnitude truncation.
        Must be positive.
    """

    k: int

    def __post_init__(self):
        if self.k <= 0:
            raise ValueError(f"k must be positive, got k={self.k}")


def topk_truncate(x: Array, k: int) -> Array:
    """Keep the k coefficients with largest absolute value; zero the rest.

    Mirror of upstream src/loss.jl:28-63. Basis-agnostic (no frequency
    layout assumption).

    Parameters
    ----------
    x : Array
        Input array of any shape.
    k : int
        Number of coefficients to keep. Clamped to `x.size` if larger.
        Returns `x` unchanged when k >= x.size.

    Returns
    -------
    Array
        Same shape and dtype as `x`; all but k entries are zero.
    """
    # `k` is a Python int (static under vmap); the size comparison is also
    # static. We branch on it OUTSIDE any traced computation.
    k2 = min(int(k), x.size)
    if k2 >= x.size:
        return x
    if k2 <= 0:
        return jnp.zeros_like(x)

    magnitudes = jnp.abs(x)
    flat = magnitudes.reshape(-1)
    # k-th largest magnitude: sort ascending, take index -k2 (i.e. k2-th from top)
    threshold = jnp.sort(flat)[-k2]

    # Tiebreaking, expressed entirely in JAX ops so this composes with vmap
    # (mirror of upstream src/loss.jl:52-60). Strictly-greater entries are
    # always kept; among the entries equal to the threshold, keep just enough
    # to reach k, in flattened-order. The `cumsum <= needed_from_ties`
    # construction selects the FIRST `needed_from_ties` ties.
    strict_mask = flat > threshold
    n_strict = jnp.sum(strict_mask.astype(jnp.int32))
    needed_from_ties = jnp.int32(k2) - n_strict
    tie_mask = (flat == threshold)
    tie_cumsum = jnp.cumsum(tie_mask.astype(jnp.int32))
    keep_tie = tie_mask & (tie_cumsum <= needed_from_ties)
    final_flat_mask = strict_mask | keep_tie
    return (x.reshape(-1) * final_flat_mask.astype(x.dtype)).reshape(x.shape)


def _apply_circuit(tensors: list[Array], code, m: int, n: int, pic: Array) -> Array:
    """Contract pic through the circuit and reshape back to (2^m, 2^n)."""
    dims = (2,) * (m + n)
    reshaped = pic.reshape(dims)
    out = code(*tensors, reshaped)
    return out.reshape(2**m, 2**n)


def _scalar_loss(
    pred: Array,
    target: Array,
    loss: AbstractLoss,
    tensors: list[Array] | None = None,
    m: int | None = None,
    n: int | None = None,
    inverse_code=None,
) -> Array:
    """Loss from already-computed forward output. Dispatches on loss type."""
    if isinstance(loss, L1Norm):
        return jnp.sum(jnp.abs(pred))
    if isinstance(loss, MSELoss):
        if inverse_code is None:
            raise ValueError("MSELoss requires inverse_code to be provided")
        truncated = topk_truncate(pred, loss.k)
        conj_tensors = [jnp.conj(t) for t in tensors]  # type: ignore[arg-type]
        reconstructed = _apply_circuit(conj_tensors, inverse_code, m, n, truncated)  # type: ignore[arg-type]
        return jnp.sum(jnp.abs(target - reconstructed) ** 2)
    raise TypeError(f"unsupported loss type: {type(loss).__name__}")


def loss_function(
    tensors: list[Array],
    m: int,
    n: int,
    code,
    pic: Array,
    loss: AbstractLoss,
    *,
    inverse_code=None,
) -> Array:
    """Compute scalar loss for a single image under the given circuit.

    Mirror of upstream src/loss.jl:94-104. See Spec Section 4.

    Parameters
    ----------
    tensors : list[Array]
        Current circuit parameters (unitary matrices, possibly phases).
    m, n : int
        Qubit counts; pic must be (2**m, 2**n).
    code : callable
        Forward einsum closure (from qft_code or equivalent).
    pic : Array
        Input image, shape (2**m, 2**n).
    loss : AbstractLoss
        L1Norm or MSELoss instance.
    inverse_code : callable, optional
        Required for MSELoss; the inverse einsum closure.
    """
    if pic.shape != (2**m, 2**n):
        raise ValueError(f"pic shape must be (2**m, 2**n) = ({2**m}, {2**n}), got {pic.shape}")
    pred = _apply_circuit(tensors, code, m, n, pic)
    return _scalar_loss(pred, pic, loss, tensors, m, n, inverse_code)
