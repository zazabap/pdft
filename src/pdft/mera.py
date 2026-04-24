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
import jax.numpy as jnp

from ._circuit import (
    HADAMARD,
    Gate,
    apply_circuit,
    compile_circuit,
    controlled_phase_diag,
)

Array = jax.Array


__all__ = ["mera_code"]


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _n_mera_gates(n_qubits: int) -> int:
    """Number of phase gates for one dim of MERA (upstream src/mera.jl:23)."""
    return 2 * (n_qubits - 1)


def _mera_single_dim_gates(
    n_qubits: int,
    qubit_offset: int,
    phases: Sequence[float],
) -> list[Gate]:
    """Build MERA gate sequence for one dimension (upstream src/mera.jl:42-73).

    For k = log2(n_qubits) layers. Each layer l has stride s = 2^(l-1).
    Disentanglers and isometries are emitted in interleaved pairs.
    """
    assert _is_pow2(n_qubits), f"n_qubits must be a power of 2, got {n_qubits}"
    assert n_qubits >= 2, f"n_qubits must be >= 2, got {n_qubits}"
    expected = _n_mera_gates(n_qubits)
    assert len(phases) == expected, f"phases length {len(phases)} != expected {expected}"

    import math

    k = int(math.log2(n_qubits))
    gates: list[Gate] = []
    phase_idx = 0

    for l in range(1, k + 1):
        s = 2 ** (l - 1)
        n_pairs = n_qubits // (2 * s)

        # Disentanglers
        for p in range(n_pairs):
            q1 = 2 * p * s + 2
            # Julia's mod1(x, n) returns ((x - 1) % n) + 1
            q2_raw = 2 * p * s + s + 2
            q2 = ((q2_raw - 1) % n_qubits) + 1
            phi = float(phases[phase_idx])
            gates.append(
                Gate(
                    kind="CP",
                    qubits=(q1 + qubit_offset, q2 + qubit_offset),
                    tensor=controlled_phase_diag(phi),
                    phase=phi,
                )
            )
            phase_idx += 1

        # Isometries
        for p in range(n_pairs):
            q1 = 2 * p * s + 1
            q2 = 2 * p * s + s + 1
            phi = float(phases[phase_idx])
            gates.append(
                Gate(
                    kind="CP",
                    qubits=(q1 + qubit_offset, q2 + qubit_offset),
                    tensor=controlled_phase_diag(phi),
                    phase=phi,
                )
            )
            phase_idx += 1

    assert phase_idx == expected
    return gates


def mera_code(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    inverse: bool = False,
) -> tuple[Callable[..., Array], list[Array], int, int]:
    """Return `(einsum_fn, initial_tensors, n_row_gates, n_col_gates)`.

    Mirror of upstream src/mera.jl:108-176. Each dimension with >= 2 qubits
    must be a power of 2; dimensions with exactly 1 qubit skip MERA in that
    direction.
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    if m >= 2 and not _is_pow2(m):
        raise ValueError(f"m must be a power of 2 when >= 2, got m={m}")
    if n >= 2 and not _is_pow2(n):
        raise ValueError(f"n must be a power of 2 when >= 2, got n={n}")

    total = m + n
    n_row_gates = _n_mera_gates(m) if m >= 2 else 0
    n_col_gates = _n_mera_gates(n) if n >= 2 else 0
    n_gates = n_row_gates + n_col_gates

    if phases is None:
        phases_list = [0.0] * n_gates
    else:
        phases_list = [float(p) for p in phases]
    if len(phases_list) != n_gates:
        raise ValueError(
            f"phases must have length {n_gates} for {m}×{n} MERA "
            f"({n_row_gates} row + {n_col_gates} col gates), got {len(phases_list)}"
        )

    gates: list[Gate] = []

    # Layer 1: Hadamards on all qubits
    for q in range(1, total + 1):
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))

    # Layer 2a: Row MERA
    if m >= 2:
        gates.extend(_mera_single_dim_gates(m, qubit_offset=0, phases=phases_list[:n_row_gates]))

    # Layer 2b: Col MERA
    if n >= 2:
        gates.extend(
            _mera_single_dim_gates(n, qubit_offset=m, phases=phases_list[n_row_gates:])
        )

    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_row_gates, n_col_gates
