"""Entangled QFT circuit construction.

Mirror of upstream src/entangled_qft.jl. Extends the standard 2D QFT by
adding `n_entangle = min(m, n)` controlled-phase gates that couple
corresponding row and column qubits, at the end of the circuit (`"back"`,
the default) or at its start (`"front"`). Upstream's `:middle` position is not
ported.
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
    phase_list,
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


# Mirrors of upstream src/entangled_qft.jl:281-326, which take the entangle gates
# to be the last `n_entangle` compact-CP tensors after the Hadamard-first sort.
# That holds for the default "back" position only. Whatever the position,
# `basis.program.tensor_indices(kind="CP", register="both")` is the entangle gates.
def get_entangle_tensor_indices(tensors: list[Array], n_entangle: int) -> list[int]:
    """Indices of the last `n_entangle` compact-CP tensors (upstream's rule)."""
    return select_last_n_cp_indices(tensors, n_entangle)


def extract_entangle_phases(tensors: list[Array], entangle_indices: list[int]) -> list[float]:
    """The phase of each compact CP tensor at `entangle_indices`."""
    return extract_phases(tensors, entangle_indices)


# Upstream's name (src/entangled_qft.jl:36-42) for the tensor of an entanglement
# gate. It is an ordinary controlled phase, in the compact 2x2 form Yao emits.
entanglement_gate = controlled_phase_diag


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
    phases = phase_list(
        entangle_phases, n_entangle, f"entangle_phases must have length min(m, n) = {n_entangle}"
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
    """Return `(code, initial_tensors, n_entangle)`; see `entangled_qft_gates`."""
    gates, n_entangle = entangled_qft_gates(
        m, n, entangle_phases=entangle_phases, entangle_position=entangle_position
    )
    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_entangle
