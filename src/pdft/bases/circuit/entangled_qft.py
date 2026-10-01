"""Entangled QFT circuit construction.

Mirror of upstream src/entangled_qft.jl. Extends the standard 2D QFT by
adding `n_entangle = min(m, n)` controlled-phase gates that couple
corresponding row and column qubits. Phase 3 supports the default
`:back` entangle_position (entanglement at the end of the circuit);
`:front` and `:middle` positions are not yet ported.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax

from ...circuit.builder import (
    Gate,
    compile_circuit,
    controlled_phase_diag,
    cp_gate,
    extract_phases,
    select_last_n_cp_indices,
)
from .qft import qft_gates

Array = jax.Array


__all__ = [
    "entangled_qft_code",
    "entangled_qft_gates",
    "entanglement_gate",
    "extract_entangle_phases",
    "get_entangle_tensor_indices",
]


# Mirrors of upstream src/entangled_qft.jl:281-326. The entangle gates are the
# last `n_entangle` compact-CP tensors after the Hadamard-first sort.
get_entangle_tensor_indices = select_last_n_cp_indices
extract_entangle_phases = extract_phases


def entanglement_gate(phi: float) -> Array:
    """2x2 tensor-network form of the 2-qubit entanglement gate.

    Mirror of upstream src/entangled_qft.jl:36-42. This is the compact
    form Yao emits: `[[1, 0], [0, exp(i*phi)]]` — NOT the full 4x4
    diagonal gate. The CP gate used in einsum contractions is the
    2x2 form from `controlled_phase_diag`, which differs: for entangled
    QFT Yao specifically emits the diagonal 2x2 `diag(1, exp(i*phi))`
    pattern, not `[[1,1],[1,exp(i*phi)]]`.

    Since `controlled_phase_diag` already matches Yao's output for
    CP gates in the yao2einsum output, we use it here too for the
    entanglement CPs.
    """
    return controlled_phase_diag(phi)


def _entangle_layer(m: int, n: int, n_entangle: int, phases: list[float]) -> list[Gate]:
    """Build the entanglement-gate layer: `n_entangle` CPs coupling row/col pairs."""
    return [cp_gate(m - k + 1, m + n - k + 1, phases[k - 1]) for k in range(1, n_entangle + 1)]


def entangled_qft_gates(
    m: int,
    n: int,
    *,
    entangle_phases: Sequence[float] | None = None,
    entangle_position: str = "back",
) -> tuple[list[Gate], int]:
    """Return `(gates, n_entangle)` for entangled 2D QFT.

    Mirror of upstream src/entangled_qft.jl:135-258. Supported positions:

      - "back" (default): QFT_row ⊗ QFT_col → Entangle
      - "front": Entangle → QFT_row ⊗ QFT_col

    The "middle" position from upstream is not yet ported (see issue #2).

    `n_entangle = min(m, n)`. Each entanglement gate k couples row qubit
    (m - k + 1) with col qubit (m + n - k + 1).
    """
    plain = qft_gates(m, n)
    if entangle_position not in ("back", "front"):
        raise ValueError(
            f"entangle_position must be 'back' or 'front', got {entangle_position!r}. "
            "'middle' is not yet ported (see GitHub issue #2)."
        )

    n_entangle = min(m, n)
    if entangle_phases is None:
        phases = [0.0] * n_entangle
    else:
        phases = [float(p) for p in entangle_phases]
    if len(phases) != n_entangle:
        raise ValueError(
            f"entangle_phases must have length min(m, n) = {n_entangle}, got {len(phases)}"
        )

    entangle = _entangle_layer(m, n, n_entangle, phases)
    return (entangle + plain if entangle_position == "front" else plain + entangle), n_entangle


def entangled_qft_code(
    m: int,
    n: int,
    *,
    entangle_phases: Sequence[float] | None = None,
    inverse: bool = False,
    entangle_position: str = "back",
) -> tuple[Callable[..., Array], list[Array], int]:
    """Return `(einsum_fn, initial_tensors, n_entangle)`; see `entangled_qft_gates`."""
    gates, n_entangle = entangled_qft_gates(
        m, n, entangle_phases=entangle_phases, entangle_position=entangle_position
    )
    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_entangle
