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

from ._circuit import (
    Gate,
    compile_circuit,
    controlled_phase_diag,
)
from .qft import _qft_gates_1d

Array = jax.Array


__all__ = [
    "entanglement_gate",
    "entangled_qft_code",
    "get_entangle_tensor_indices",
    "extract_entangle_phases",
]


def get_entangle_tensor_indices(tensors: list[Array], n_entangle: int) -> list[int]:
    """Indices of the entanglement-gate tensors in `tensors`.

    Mirror of upstream src/entangled_qft.jl:281-313. Entangle gates are the
    last `n_entangle` compact-CP tensors after the Hadamard-first sort.
    """
    from ._circuit import select_last_n_cp_indices

    return select_last_n_cp_indices(tensors, n_entangle)


def extract_entangle_phases(tensors: list[Array], entangle_indices: list[int]) -> list[float]:
    """Extract phases φ_k from entanglement-gate tensors.

    Mirror of upstream src/entangled_qft.jl:316-326.
    """
    from ._circuit import extract_phase_from_cp

    return [extract_phase_from_cp(tensors[idx]) for idx in entangle_indices]


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
    gates: list[Gate] = []
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
    return gates


def entangled_qft_code(
    m: int,
    n: int,
    *,
    entangle_phases: Sequence[float] | None = None,
    inverse: bool = False,
    entangle_position: str = "back",
) -> tuple[Callable[..., Array], list[Array], int]:
    """Return `(einsum_fn, initial_tensors, n_entangle)` for entangled 2D QFT.

    Mirror of upstream src/entangled_qft.jl:135-258. Supported positions:

      - "back" (default): QFT_row ⊗ QFT_col → Entangle
      - "front": Entangle → QFT_row ⊗ QFT_col

    The "middle" position from upstream is not yet ported (see issue #2).

    `n_entangle = min(m, n)`. Each entanglement gate k couples row qubit
    (m - k + 1) with col qubit (m + n - k + 1).
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
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

    qft_gates = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    entangle_gates = _entangle_layer(m, n, n_entangle, phases)

    if entangle_position == "front":
        gates = entangle_gates + qft_gates
    else:  # "back"
        gates = qft_gates + entangle_gates

    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_entangle
