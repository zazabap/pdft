"""The program object: a circuit's structure as hashable data, and the applier keyed on it."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import pdft
from pdft.bases.circuit.qft import _qft_gates_1d
from pdft.circuit.builder import (
    HADAMARD,
    CircuitCode,
    Gate,
    Program,
    _run,
    compile_circuit,
    compile_program,
    controlled_phase_diag,
    sorted_gate_program,
    u4_from_phase,
)


def _gates() -> list[Gate]:
    """Deliberately not Hadamard-first, so the stored order differs from the temporal one."""
    return [
        Gate(kind="U4", qubits=(1, 2), tensor=u4_from_phase(0.3), phase=0.3),
        Gate(kind="H", qubits=(1,), tensor=HADAMARD, phase=0.0),
        Gate(kind="CP", qubits=(2, 1), tensor=controlled_phase_diag(0.7), phase=0.7),
        Gate(kind="H", qubits=(2,), tensor=HADAMARD, phase=0.0),
    ]


def test_compile_program_keeps_temporal_steps_and_stores_hadamards_first():
    program, tensors = compile_program(_gates(), 1, 1)
    assert (program.m, program.n) == (1, 1)
    assert [kind for kind, _ in program.steps] == ["U4", "H", "CP", "H"]
    assert program.slot == (2, 0, 3, 1)
    assert [t.shape for t in tensors] == [(2, 2), (2, 2), (2, 2, 2, 2), (2, 2)]
    # step i reads the tensor it was emitted with
    for step, gate in enumerate(_gates()):
        assert jnp.array_equal(tensors[program.slot[step]], gate["tensor"])


def test_sorted_steps_is_the_stored_order():
    gates = _qft_gates_1d(3, 0) + _qft_gates_1d(2, 3)
    program, tensors = compile_program(gates, 3, 2)
    assert list(program.sorted_steps) == sorted_gate_program(gates)
    assert len(program.sorted_steps) == len(tensors)


def test_a_program_is_a_value():
    a, _ = compile_program(_gates(), 1, 1)
    b, _ = compile_program(_gates(), 1, 1)
    assert a == b and hash(a) == hash(b) and a is not b
    assert a != compile_program(_gates(), 2, 0)[0]
    assert isinstance(a, Program) and {a: 1}[b] == 1


def test_code_compares_by_program_and_direction():
    forward, _ = compile_circuit(_gates(), 1, 1, inverse=False)
    again, _ = compile_circuit(_gates(), 1, 1, inverse=False)
    inverse, _ = compile_circuit(_gates(), 1, 1, inverse=True)
    assert isinstance(forward, CircuitCode)
    assert forward == again and hash(forward) == hash(again)
    assert forward != inverse and forward.program == inverse.program


def test_two_instances_of_a_basis_share_structure_and_compiled_code():
    a, b = pdft.QFTBasis(m=2, n=3), pdft.QFTBasis(m=2, n=3)
    assert a.code == b.code and a.inv_code == b.inv_code
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((4, 8)))
    a.forward_transform(x)
    compiled = _run._cache_size()
    b.forward_transform(x)
    assert _run._cache_size() == compiled


def test_inverse_walks_the_steps_backwards_with_swapped_legs():
    """With conjugated tensors the inverse code is the adjoint, so for unitary
    gates it undoes the forward code; with the forward order it would not."""
    rng = np.random.default_rng(1)
    pic = jnp.asarray(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
    forward, tensors = compile_circuit(_gates(), 1, 1, inverse=False)
    inverse, _ = compile_circuit(_gates(), 1, 1, inverse=True)
    out = forward(*tensors, pic)
    back = inverse(*[jnp.conj(t) for t in tensors], out)
    np.testing.assert_allclose(back, pic, atol=1e-12)
    assert not jnp.allclose(forward(*[jnp.conj(t) for t in tensors], out), pic, atol=1e-6)
