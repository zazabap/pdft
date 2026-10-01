"""freeze_as_blocked: reduce a full circuit to its blocked equivalent.

Resets the outer (block-index) gates of a QFT/Rich/RealRich circuit to
identity and returns the frozen tensor indices, so that

    train_basis_batched(frozen_basis, frozen_indices=frozen, ...)

reproduces BlockedBasis(inner, block_log_m, block_log_n) training dynamics.

Gradient-norm clipping note: frozen slots have their Euclidean gradient zeroed
before manifold projection, so their Riemannian gradient is zero and they
contribute zero to the global grad-norm. The clip factor therefore matches
BlockedBasis exactly — parity holds with max_grad_norm set (see
test_frozen_qft_outer_gates_matches_blocked_qft_training at max_grad_norm=1.0).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ...circuit.builder import identity_tensor
from ..core import with_tensors

Array = jax.Array

__all__ = ["freeze_as_blocked"]


def freeze_as_blocked(basis: Any, block_log_m: int, block_log_n: int) -> tuple[Any, list[int]]:
    """Return ``(basis_copy_with_identity_outer_gates, frozen_indices)``.

    Training the returned basis with ``frozen_indices`` reproduces
    ``BlockedBasis(inner, block_log_m, block_log_n)`` training dynamics, where
    ``inner`` is the same basis type at ``(m - block_log_m, n - block_log_n)``.
    The outer (block-index) gates are reset to identity; inner gates keep the
    input basis's tensor values.

    Supports ``QFTBasis``, ``RichBasis``, ``RealRichBasis``: the bases that
    declare ``freezes_to_blocked``, where the gates on the kept qubits of a
    register are the whole circuit of a smaller register. The gates and the
    qubits they touch are read from the basis's own program.

    ``block_log_m == block_log_n == 0`` is a valid no-op: no qubits are
    block-index qubits, so nothing is frozen and ``frozen_indices`` is empty
    (the returned basis is a tensor-copy of the input).
    """
    btype = type(basis)
    if not getattr(btype, "freezes_to_blocked", False):
        raise TypeError(
            f"freeze_as_blocked supports QFTBasis, RichBasis, RealRichBasis; got {btype.__name__}"
        )
    if block_log_m < 0 or block_log_n < 0:
        raise ValueError(
            "block_log_m and block_log_n must be >= 0; got "
            f"block_log_m={block_log_m}, block_log_n={block_log_n}"
        )
    m, n = basis.m, basis.n
    if m - block_log_m < 1 or n - block_log_n < 1:
        raise ValueError(
            f"block partition leaves no inner qubits: m={m}, n={n}, "
            f"block_log_m={block_log_m}, block_log_n={block_log_n} "
            "(need m - block_log_m >= 1 and n - block_log_n >= 1)"
        )

    # Block-index qubits: higher-numbered qubits per dim (Yao little-endian),
    # matching BlockedBasis.
    block_qubits = set(range(m - block_log_m + 1, m + 1)) | set(
        range(m + n - block_log_n + 1, m + n + 1)
    )

    stored = basis.program.sorted_steps
    if len(stored) != len(basis.tensors):
        raise AssertionError(
            f"gate program length {len(stored)} != tensor count {len(basis.tensors)}"
        )

    new_tensors = [jnp.array(t, copy=True) for t in basis.tensors]
    frozen_indices: list[int] = []
    for i, (kind, qubits) in enumerate(stored):
        if set(qubits) & block_qubits:
            new_tensors[i] = identity_tensor(kind).astype(new_tensors[i].dtype)
            frozen_indices.append(i)

    return with_tensors(basis, new_tensors), frozen_indices
