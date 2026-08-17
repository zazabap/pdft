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
    HADAMARD,
    Gate,
    compile_circuit,
    controlled_phase_diag,
    u4_from_phase,
)

Array = jax.Array


__all__ = [
    "extract_tebd_phases",
    "get_tebd_gate_indices",
    "tebd_code",
]


def get_tebd_gate_indices(tensors: list[Array], n_gates: int) -> list[int]:
    """Indices of TEBD gate tensors. Mirror of upstream src/tebd.jl:124-140."""
    from ...circuit.builder import select_last_n_cp_indices

    return select_last_n_cp_indices(tensors, n_gates)


def extract_tebd_phases(tensors: list[Array], gate_indices: list[int]) -> list[float]:
    """Mirror of upstream src/tebd.jl:154-160."""
    from ...circuit.builder import extract_phase_from_cp

    return [extract_phase_from_cp(tensors[idx]) for idx in gate_indices]


def tebd_code(
    m: int,
    n: int,
    *,
    phases: Sequence[float] | None = None,
    inverse: bool = False,
    parametrization: str = "cp",
) -> tuple[Callable[..., Array], list[Array], int, int]:
    """Return `(einsum_fn, initial_tensors, n_row_gates, n_col_gates)`.

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
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    if parametrization not in ("cp", "u4"):
        raise ValueError(
            f"parametrization must be 'cp' or 'u4', got {parametrization!r}"
        )

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

    # One ring gate. "cp" keeps the compact diagonal tensor; "u4" emits the
    # dense two-qubit tensor warm-started at the same operator.
    def _ring_gate(q_ctrl: int, q_tgt: int, phi: float) -> Gate:
        if parametrization == "u4":
            return Gate(
                kind="U4",
                qubits=(q_ctrl, q_tgt),
                tensor=u4_from_phase(phi),
                phase=phi,
            )
        return Gate(
            kind="CP",
            qubits=(q_ctrl, q_tgt),
            tensor=controlled_phase_diag(phi),
            phase=phi,
        )

    # Layer 1: Hadamards on all qubits
    for q in range(1, total + 1):
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))

    gate_idx = 0

    # Layer 2a: Row ring — (i, i+1) for i=1..m-1
    for i in range(1, m):
        gates.append(_ring_gate(i, i + 1, phases_list[gate_idx]))
        gate_idx += 1
    # Wrap-around: (m, 1)
    gates.append(_ring_gate(m, 1, phases_list[gate_idx]))
    gate_idx += 1

    # Layer 2b: Col ring — (m+i, m+i+1) for i=1..n-1
    for i in range(1, n):
        gates.append(_ring_gate(m + i, m + i + 1, phases_list[gate_idx]))
        gate_idx += 1
    # Wrap-around: (m+n, m+1)
    gates.append(_ring_gate(m + n, m + 1, phases_list[gate_idx]))
    gate_idx += 1

    assert gate_idx == n_gates

    code, tensors = compile_circuit(gates, m, n, inverse=inverse)
    return code, tensors, n_row_gates, n_col_gates
