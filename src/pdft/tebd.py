"""TEBD (Time-Evolving Block Decimation) circuit with 2D ring topology.

Mirror of upstream src/tebd.jl. Layer 1: Hadamard on all m+n qubits.
Layer 2: two rings of controlled-phase gates (row ring has m gates
including the wrap-around; col ring has n gates).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import jax

from ._circuit import (
    HADAMARD,
    Gate,
    compile_circuit,
    controlled_phase_diag,
)

Array = jax.Array


__all__ = ["tebd_code"]


def tebd_code(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    inverse: bool = False,
) -> tuple[Callable[..., Array], list[Array], int, int]:
    """Return `(einsum_fn, initial_tensors, n_row_gates, n_col_gates)`.

    Mirror of upstream src/tebd.jl:48-110.

    The circuit is:
        1. H on each of the m+n qubits.
        2. Row ring: CP(i, i+1) for i=1..m-1, then wrap-around CP(m, 1).
        3. Col ring: CP(m+i, m+i+1) for i=1..n-1, then wrap-around CP(m+n, m+1).
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")

    n_row_gates = m
    n_col_gates = n
    n_gates = n_row_gates + n_col_gates

    if phases is None:
        phases_list = [0.0] * n_gates
    else:
        phases_list = [float(p) for p in phases]
    if len(phases_list) != n_gates:
        raise ValueError(
            f"phases must have length n_row_gates + n_col_gates = {n_gates}, got {len(phases_list)}"
        )

    gates: list[Gate] = []
    total = m + n

    # Layer 1: Hadamards on all qubits
    for q in range(1, total + 1):
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))

    gate_idx = 0

    # Layer 2a: Row ring — CP(i, i+1) for i=1..m-1
    for i in range(1, m):
        phi = phases_list[gate_idx]
        gates.append(
            Gate(
                kind="CP",
                qubits=(i, i + 1),
                tensor=controlled_phase_diag(phi),
                phase=phi,
            )
        )
        gate_idx += 1
    # Wrap-around: CP(m, 1)
    phi = phases_list[gate_idx]
    gates.append(
        Gate(kind="CP", qubits=(m, 1), tensor=controlled_phase_diag(phi), phase=phi)
    )
    gate_idx += 1

    # Layer 2b: Col ring — CP(m+i, m+i+1) for i=1..n-1
    for i in range(1, n):
        phi = phases_list[gate_idx]
        gates.append(
            Gate(
                kind="CP",
                qubits=(m + i, m + i + 1),
                tensor=controlled_phase_diag(phi),
                phase=phi,
            )
        )
        gate_idx += 1
    # Wrap-around: CP(m+n, m+1)
    phi = phases_list[gate_idx]
    gates.append(
        Gate(
            kind="CP",
            qubits=(m + n, m + 1),
            tensor=controlled_phase_diag(phi),
            phase=phi,
        )
    )
    gate_idx += 1

    assert gate_idx == n_gates

    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_row_gates, n_col_gates
