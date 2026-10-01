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
    compile_circuit,
    extract_phases,
    hadamards_then_layers,
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


def _n_tebd_gates(n_qubits: int) -> int:
    """Number of gates in the ring of one register: one per qubit, the wrap-around included."""
    return n_qubits


def _ring(
    n_qubits: int, offset: int, phases: Sequence[float], gate: Callable[[int, int, float], Gate]
) -> list[Gate]:
    """The ring of one register: ``(i, i+1)`` for ``i = 1..n-1``, then the wrap-around ``(n, 1)``.

    Both cases are the pair ``(i, i mod n + 1)``, one per phase.
    """
    return [
        gate(offset + i, offset + i % n_qubits + 1, phi) for i, phi in enumerate(phases, start=1)
    ]


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

    Step 1, the split of ``phases`` between the two rings and the choice of
    gate form are ``hadamards_then_layers``, which MERA uses too. What is
    TEBD's own is the ring of one register, ``_ring``.
    """
    return hadamards_then_layers(_ring, _n_tebd_gates, m, n, phases, parametrization)


def tebd_code(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    inverse: bool = False,
    parametrization: str = "cp",
) -> tuple[Callable[..., Array], list[Array], int, int]:
    """Return `(code, initial_tensors, n_row_gates, n_col_gates)`; see `tebd_gates`."""
    gates, n_row_gates, n_col_gates = tebd_gates(
        m, n, phases=phases, parametrization=parametrization
    )
    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_row_gates, n_col_gates
