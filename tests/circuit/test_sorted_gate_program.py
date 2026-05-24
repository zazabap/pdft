"""Unit test for builder.sorted_gate_program (Hadamard-first gate metadata)."""

from __future__ import annotations

import pdft  # noqa: F401  (enables x64; keeps import side effects)
from pdft.bases.circuit.qft import _qft_gates_1d
from pdft.circuit.builder import sorted_gate_program


def test_sorted_gate_program_hadamards_first():
    gates = _qft_gates_1d(3, 0) + _qft_gates_1d(3, 3)
    prog = sorted_gate_program(gates)

    kinds = [k for k, q in prog]
    assert kinds == ["H"] * 6 + ["CP"] * 6

    h_qubits = [q for k, q in prog[:6]]
    assert h_qubits == [(1,), (2,), (3,), (4,), (5,), (6,)]

    cp_qubits = [q for k, q in prog[6:]]
    assert cp_qubits == [(2, 1), (3, 1), (3, 2), (5, 4), (6, 4), (6, 5)]


def test_sorted_gate_program_len_matches_gates():
    gates = _qft_gates_1d(2, 0) + _qft_gates_1d(2, 2)
    assert len(sorted_gate_program(gates)) == len(gates)
