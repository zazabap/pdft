"""freeze_as_blocked: reduce a full circuit to its blocked equivalent.

Resets the outer (block-index) gates of a QFT/Rich/RealRich circuit to
identity and returns the frozen tensor indices, so that

    train_basis_batched(frozen_basis, frozen_indices=frozen, ...)

reproduces BlockedBasis(inner, block_log_m, block_log_n) training dynamics.
See docs/superpowers/specs/2026-05-24-circuit-rich-frozen-blocked-design.md.

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

from ...circuit.builder import controlled_phase_diag, sorted_gate_program

Array = jax.Array

__all__ = ["freeze_as_blocked"]


def _identity_for_kind(kind: str) -> Array:
    """Identity-acting tensor for a gate kind, as complex128."""
    if kind == "H":
        return jnp.eye(2, dtype=jnp.complex128)
    if kind == "U4":
        return jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    if kind == "CP":
        return controlled_phase_diag(0.0)  # diag(1,1,1,1) in compact 2x2 form
    raise AssertionError(f"unknown gate kind: {kind}")


def freeze_as_blocked(basis: Any, block_log_m: int, block_log_n: int) -> tuple[Any, list[int]]:
    """Return ``(basis_copy_with_identity_outer_gates, frozen_indices)``.

    Training the returned basis with ``frozen_indices`` reproduces
    ``BlockedBasis(inner, block_log_m, block_log_n)`` training dynamics, where
    ``inner`` is the same basis type at ``(m - block_log_m, n - block_log_n)``.
    The outer (block-index) gates are reset to identity; inner gates keep the
    input basis's tensor values.

    Supports ``QFTBasis``, ``RichBasis``, ``RealRichBasis``.

    ``block_log_m == block_log_n == 0`` is a valid no-op: no qubits are
    block-index qubits, so nothing is frozen and ``frozen_indices`` is empty
    (the returned basis is a tensor-copy of the input).
    """
    # Lazy imports avoid a real bases.base <-> bases.circuit import cycle.
    # base.py runs `from .circuit.qft import qft_code` at module load; importing
    # that submodule first executes bases/circuit/__init__.py, which imports
    # THIS module. A module-top `from ..base import QFTBasis` would then touch
    # base.py while it is still partially initialised (before QFTBasis exists)
    # and raise ImportError. Verified: hoisting these to the top breaks
    # `import pdft`. Keep them inside the function.
    from ..base import QFTBasis
    from .qft import _qft_gates_1d
    from .real_rich import RealRichBasis, _real_rich_qft_gates_1d
    from .rich import RichBasis, _rich_qft_gates_1d

    gate_builders = {
        QFTBasis: _qft_gates_1d,
        RichBasis: _rich_qft_gates_1d,
        RealRichBasis: _real_rich_qft_gates_1d,
    }
    btype = type(basis)
    if btype not in gate_builders:
        raise TypeError(
            "freeze_as_blocked supports QFTBasis, RichBasis, RealRichBasis; "
            f"got {btype.__name__}"
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

    builder = gate_builders[btype]
    gates = builder(m, offset=0) + builder(n, offset=m)
    program = sorted_gate_program(gates)
    if len(program) != len(basis.tensors):
        raise AssertionError(
            f"gate program length {len(program)} != tensor count {len(basis.tensors)}"
        )

    new_tensors = [jnp.array(t, copy=True) for t in basis.tensors]
    frozen_indices: list[int] = []
    for i, (kind, qubits) in enumerate(program):
        if set(qubits) & block_qubits:
            new_tensors[i] = _identity_for_kind(kind).astype(new_tensors[i].dtype)
            frozen_indices.append(i)

    new_basis = btype(m=m, n=n, tensors=new_tensors, code=basis.code, inv_code=basis.inv_code)
    return new_basis, frozen_indices
