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
import jax.numpy as jnp

from ._circuit import (
    Gate,
    apply_circuit,
    compile_circuit,
    controlled_phase_diag,
)
from .qft import _qft_gates_1d

Array = jax.Array


__all__ = [
    "entanglement_gate",
    "entangled_qft_code",
]


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


def entangled_qft_code(
    m: int,
    n: int,
    *,
    entangle_phases: Sequence[float] | None = None,
    inverse: bool = False,
) -> tuple[Callable[..., Array], list[Array], int]:
    """Return `(einsum_fn, initial_tensors, n_entangle)` for entangled 2D QFT.

    Phase 3 scope: `:back` position only. The entanglement gates are appended
    after both row-QFT and col-QFT. This matches upstream's default
    `entangle_position=:back`.

    `n_entangle = min(m, n)`. Each entanglement gate k (1..n_entangle) couples
    row qubit (m - k + 1) with col qubit (m + n - k + 1). When phases is
    None it defaults to zeros — which makes the entanglement gate the
    identity (equivalent to standard QFT).
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")

    n_entangle = min(m, n)
    if entangle_phases is None:
        phases = [0.0] * n_entangle
    else:
        phases = [float(p) for p in entangle_phases]
    if len(phases) != n_entangle:
        raise ValueError(
            f"entangle_phases must have length min(m, n) = {n_entangle}, got {len(phases)}"
        )

    gates: list[Gate] = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    # Append entanglement layer at :back
    for k in range(1, n_entangle + 1):
        x_qubit = m - k + 1
        y_qubit = m + n - k + 1
        phi = phases[k - 1]
        gates.append(
            Gate(
                kind="CP",
                qubits=(x_qubit, y_qubit),
                tensor=controlled_phase_diag(phi),
                phase=phi,
            )
        )

    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_entangle
