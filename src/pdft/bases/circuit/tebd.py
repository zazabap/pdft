"""TEBD (Time-Evolving Block Decimation) circuit with 2D ring topology.

Mirror of upstream src/tebd.jl. Layer 1: Hadamard on all m+n qubits.
Layer 2: two rings of two-qubit gates (row ring has m gates including the
wrap-around; col ring has n gates).

``parametrization`` selects the ring-gate family:

* ``"cp"`` (default, the historical behaviour) stores each ring gate as a
  compact ``(2, 2)`` diagonal controlled phase trained on ``U(1)^4``.
* ``"u4"`` stores each ring gate as a dense ``(2, 2, 2, 2)`` two-qubit
  tensor trained on ``U(4)`` — the canonical TEBD gate, since a Trotter
  step ``exp(-i h_{i,i+1} tau)`` of a two-site Hamiltonian is a *general*
  two-qubit unitary, not a diagonal phase. Initialised via
  ``u4_from_phase`` so that, at the same ``phases``, the ``"u4"`` circuit
  starts at exactly the same operator as the ``"cp"`` circuit; only the
  manifold it relaxes over is larger.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax

from ...circuit.builder import (
    Gate,
    check_qubits,
    compile_circuit,
    extract_phases,
    hadamard_gate,
    phase_gate,
    select_last_n_cp_indices,
)

Array = jax.Array


__all__ = [
    "extract_tebd_phases",
    "get_tebd_gate_indices",
    "tebd_code",
    "tebd_gates",
]


# Mirrors of upstream src/tebd.jl:124-160.
get_tebd_gate_indices = select_last_n_cp_indices
extract_tebd_phases = extract_phases


def tebd_gates(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    parametrization: str = "cp",
) -> tuple[list[Gate], int, int]:
    """Return `(gates, n_row_gates, n_col_gates)`.

    Mirror of upstream src/tebd.jl:48-110.

    The circuit is:
        1. H on each of the m+n qubits.
        2. Row ring: ring gate (i, i+1) for i=1..m-1, then wrap-around (m, 1).
        3. Col ring: ring gate (m+i, m+i+1) for i=1..n-1, then wrap-around
           (m+n, m+1).

    ``parametrization`` is ``"cp"`` (diagonal, ``U(1)^4``) or ``"u4"``
    (dense two-qubit, ``U(4)`` — the canonical TEBD gate). Both start from
    the same operator for a given ``phases``; see the module docstring.
    """
    check_qubits(m, n)
    ring_gate = phase_gate(parametrization)

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

    # Layer 1: Hadamards on all qubits
    gates = [hadamard_gate(q) for q in range(1, m + n + 1)]

    gate_idx = 0

    # Layer 2a: Row ring — (i, i+1) for i=1..m-1
    for i in range(1, m):
        gates.append(ring_gate(i, i + 1, phases_list[gate_idx]))
        gate_idx += 1
    # Wrap-around: (m, 1)
    gates.append(ring_gate(m, 1, phases_list[gate_idx]))
    gate_idx += 1

    # Layer 2b: Col ring — (m+i, m+i+1) for i=1..n-1
    for i in range(1, n):
        gates.append(ring_gate(m + i, m + i + 1, phases_list[gate_idx]))
        gate_idx += 1
    # Wrap-around: (m+n, m+1)
    gates.append(ring_gate(m + n, m + 1, phases_list[gate_idx]))
    gate_idx += 1

    assert gate_idx == n_gates
    return gates, n_row_gates, n_col_gates


def tebd_code(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    inverse: bool = False,
    parametrization: str = "cp",
) -> tuple[Callable[..., Array], list[Array], int, int]:
    """Return `(einsum_fn, initial_tensors, n_row_gates, n_col_gates)`; see `tebd_gates`."""
    gates, n_row_gates, n_col_gates = tebd_gates(
        m, n, phases=phases, parametrization=parametrization
    )
    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_row_gates, n_col_gates
