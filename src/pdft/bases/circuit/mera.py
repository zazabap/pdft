"""MERA (Multi-scale Entanglement Renormalization Ansatz) circuit.

Mirror of upstream src/mera.jl. Layer 1: Hadamard on all qubits. Layer 2:
two hierarchical MERA structures (disentanglers + isometries), one per
dimension, each with `log2(n_qubits)` levels and `2*(n_qubits-1)` gates.
Each dimension requires a power-of-2 qubit count (or 1 for no MERA in
that dimension).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax

from ...circuit.builder import (
    Gate,
    compile_circuit,
    extract_phases,
    hadamards_then_layers,
    select_last_n_cp_indices,
)

Array = jax.Array


__all__ = [
    "extract_mera_phases",
    "get_mera_gate_indices",
    "mera_code",
    "mera_gates",
]


# Mirrors of upstream src/mera.jl:190-226.
get_mera_gate_indices = select_last_n_cp_indices
extract_mera_phases = extract_phases


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _n_mera_gates(n_qubits: int) -> int:
    """Number of phase gates for one dim of MERA (upstream src/mera.jl:23); none on one qubit."""
    return 2 * (n_qubits - 1)


def _mera_single_dim_gates(
    n_qubits: int,
    qubit_offset: int,
    phases: Sequence[float],
    pair_gate: Callable[[int, int, float], Gate],
) -> list[Gate]:
    """Build MERA gate sequence for one dimension (upstream src/mera.jl:42-73).

    For k = log2(n_qubits) layers. Each layer l has stride s = 2^(l-1).
    Disentanglers and isometries are emitted in interleaved pairs. A
    one-qubit register has no layers and gets no gates.

    ``pair_gate(q_ctrl, q_tgt, phi)`` builds each of them: diagonal on
    ``U(1)^4``, or dense on ``U(4)``, the canonical family since a MERA
    disentangler/isometry is a general two-site unitary. Both start at the
    same operator for a given ``phases``.
    """
    assert _is_pow2(n_qubits), f"n_qubits must be a power of 2, got {n_qubits}"
    expected = _n_mera_gates(n_qubits)
    assert len(phases) == expected, f"phases length {len(phases)} != expected {expected}"

    import math

    k = int(math.log2(n_qubits))
    gates: list[Gate] = []
    phase_idx = 0

    for layer in range(1, k + 1):
        s = 2 ** (layer - 1)
        n_pairs = n_qubits // (2 * s)

        # Disentanglers
        for p in range(n_pairs):
            q1 = 2 * p * s + 2
            # Julia's mod1(x, n) returns ((x - 1) % n) + 1
            q2_raw = 2 * p * s + s + 2
            q2 = ((q2_raw - 1) % n_qubits) + 1
            gates.append(pair_gate(q1 + qubit_offset, q2 + qubit_offset, float(phases[phase_idx])))
            phase_idx += 1

        # Isometries
        for p in range(n_pairs):
            q1 = 2 * p * s + 1
            q2 = 2 * p * s + s + 1
            gates.append(pair_gate(q1 + qubit_offset, q2 + qubit_offset, float(phases[phase_idx])))
            phase_idx += 1

    assert phase_idx == expected
    return gates


def mera_gates(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    parametrization: str = "cp",
) -> tuple[list[Gate], int, int]:
    """Return `(gates, n_row_gates, n_col_gates)`.

    Mirror of upstream src/mera.jl:108-176. Each dimension with >= 2 qubits
    must be a power of 2; dimensions with exactly 1 qubit skip MERA in that
    direction.

    ``parametrization`` is ``"cp"`` (diagonal, ``U(1)^4``) or ``"u4"``
    (dense two-qubit, ``U(4)`` — the canonical disentangler/isometry).
    """
    for name, n_qubits in (("m", m), ("n", n)):
        if n_qubits >= 2 and not _is_pow2(n_qubits):
            raise ValueError(f"{name} must be a power of 2 when >= 2, got {name}={n_qubits}")
    return hadamards_then_layers(
        _mera_single_dim_gates, _n_mera_gates, m, n, phases, parametrization
    )


def mera_code(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    inverse: bool = False,
    parametrization: str = "cp",
) -> tuple[Callable[..., Array], list[Array], int, int]:
    """Return `(code, initial_tensors, n_row_gates, n_col_gates)`; see `mera_gates`."""
    gates, n_row_gates, n_col_gates = mera_gates(
        m, n, phases=phases, parametrization=parametrization
    )
    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_row_gates, n_col_gates
