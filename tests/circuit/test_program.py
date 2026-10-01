"""`Program`: a circuit's structure as hashable data, and the questions it answers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import pdft
from pdft.bases.circuit.qft import qft_gates_1d
from pdft.circuit import REGISTERS
from pdft.circuit.builder import (
    CircuitCode,
    Gate,
    Program,
    _run,
    compile_circuit,
    compile_program,
    u4_from_phase,
)

from ..helpers import complex_image, small_circuit


def test_compile_program_keeps_temporal_steps_and_stores_hadamards_first():
    program, tensors = compile_program(small_circuit(), 1, 1)
    assert (program.m, program.n) == (1, 1)
    assert [kind for kind, _ in program.steps] == ["U4", "H", "CP", "H"]
    assert program.slot == (2, 0, 3, 1)
    assert [t.shape for t in tensors] == [(2, 2), (2, 2), (2, 2, 2, 2), (2, 2)]
    # step i reads the tensor it was emitted with
    for step, gate in enumerate(small_circuit()):
        assert jnp.array_equal(tensors[program.slot[step]], gate["tensor"])


def test_sorted_steps_is_the_stored_order():
    gates = qft_gates_1d(3, 0) + qft_gates_1d(3, 3)
    program, tensors = compile_program(gates, 3, 3)
    stored = program.sorted_steps
    assert len(stored) == len(tensors) == len(gates)
    assert [kind for kind, _ in stored] == ["H"] * 6 + ["CP"] * 6
    assert [qubits for _, qubits in stored[:6]] == [(1,), (2,), (3,), (4,), (5,), (6,)]
    assert [qubits for _, qubits in stored[6:]] == [(2, 1), (3, 1), (3, 2), (5, 4), (6, 4), (6, 5)]
    # the stored order is the temporal one, read through ``slot``
    assert all(stored[slot] == step for step, slot in zip(program.steps, program.slot))


def test_a_program_is_a_value():
    a, _ = compile_program(small_circuit(), 1, 1)
    b, _ = compile_program(small_circuit(), 1, 1)
    assert a == b and hash(a) == hash(b) and a is not b
    assert a != compile_program(small_circuit(), 2, 0)[0]
    assert isinstance(a, Program) and {a: 1}[b] == 1


def test_code_compares_by_program_and_direction():
    forward, _ = compile_circuit(small_circuit(), 1, 1, inverse=False)
    again, _ = compile_circuit(small_circuit(), 1, 1, inverse=False)
    inverse, _ = compile_circuit(small_circuit(), 1, 1, inverse=True)
    assert isinstance(forward, CircuitCode)
    assert forward == again and hash(forward) == hash(again)
    assert forward != inverse and forward.program == inverse.program


def test_tensor_indices_by_kind_on_the_qft():
    program = pdft.QFTBasis(m=2, n=3).program
    assert program.tensor_indices() == list(range(9))
    assert program.tensor_indices(kind="H") == [0, 1, 2, 3, 4]
    assert program.tensor_indices(kind="CP") == [5, 6, 7, 8]
    assert program.tensor_indices(kind="U4") == program.tensor_indices(kind="CRY") == []


def test_tensor_indices_by_register():
    program = pdft.EntangledQFTBasis(m=2, n=3).program
    row, column, both = (program.tensor_indices(register=r) for r in REGISTERS)
    # 2 + 3 Hadamards, 1 + 3 QFT phases, min(m, n) = 2 entanglers across the registers
    assert (len(row), len(column), len(both)) == (3, 6, 2)
    assert sorted(row + column + both) == list(range(11))
    assert program.tensor_indices(kind="H", register="row") == [0, 1]
    assert program.tensor_indices(kind="H", register="both") == []
    stored = program.sorted_steps
    assert all(max(stored[i][1]) <= 2 for i in row)
    assert all(min(stored[i][1]) > 2 for i in column)
    assert all(min(stored[i][1]) <= 2 < max(stored[i][1]) for i in both)


def test_tensor_indices_refuses_an_unknown_kind_or_register():
    """A typo must not silently freeze nothing."""
    program = pdft.QFTBasis(m=2, n=2).program
    with pytest.raises(ValueError, match="kind must be one of"):
        program.tensor_indices(kind="h")
    with pytest.raises(ValueError, match="register must be one of"):
        program.tensor_indices(register="rows")


def test_two_instances_of_a_basis_share_one_compiled_applier():
    """Equal codes, and so one entry in the jit cache for both."""
    a, b = pdft.QFTBasis(m=2, n=3), pdft.QFTBasis(m=2, n=3)
    x = complex_image((4, 8))
    a.forward_transform(x)
    compiled = _run._cache_size()
    b.forward_transform(x)
    assert a.code == b.code and _run._cache_size() == compiled


def test_compile_program_refuses_a_tensor_of_the_wrong_shape():
    gates = small_circuit()
    gates[2] = Gate(kind="CP", qubits=(2, 1), tensor=u4_from_phase(0.7), phase=0.7)
    with pytest.raises(ValueError, match=r"gate 2 of kind 'CP' needs a tensor of shape \(2, 2\)"):
        compile_program(gates, 1, 1)
