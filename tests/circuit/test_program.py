"""The program object: a circuit's structure as hashable data, and the applier keyed on it."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.circuit.qft import _qft_gates_1d
from pdft.circuit.builder import (
    HADAMARD,
    CircuitCode,
    Gate,
    Program,
    _run,
    apply_circuit,
    apply_program,
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


def _image(rng, shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def test_apply_program_is_apply_circuit_on_one_double_precision_image():
    program, tensors = compile_program(_gates(), 1, 1)
    x = _image(np.random.default_rng(2), (2, 2))
    code = CircuitCode(program)
    assert jnp.array_equal(
        apply_program(program, tensors, x), apply_circuit(tensors, code, 1, 1, x)
    )


def test_apply_program_carries_leading_axes():
    gates = _qft_gates_1d(2, 0) + _qft_gates_1d(3, 2)
    program, tensors = compile_program(gates, 2, 3)
    stack = _image(np.random.default_rng(3), (2, 5, 4, 8))
    out = apply_program(program, tensors, stack)
    assert out.shape == stack.shape
    for i in range(2):
        for j in range(5):
            assert jnp.array_equal(out[i, j], apply_program(program, tensors, stack[i, j]))


def test_apply_program_follows_the_precision_of_the_image():
    program, tensors = compile_program(_gates(), 1, 1)
    x = jnp.asarray(np.random.default_rng(4).random((2, 2)))
    double = apply_program(program, tensors, x)
    single = apply_program(program, tensors, x.astype(jnp.float32))
    assert double.dtype == jnp.complex128 and single.dtype == jnp.complex64
    assert apply_program(program, tensors, x.astype(jnp.complex64)).dtype == jnp.complex64
    np.testing.assert_allclose(single, double, atol=1e-6)


def test_apply_program_refuses_another_image_size():
    program, tensors = compile_program(_gates(), 1, 1)
    with pytest.raises(ValueError, match="image shape"):
        apply_program(program, tensors, jnp.zeros((4, 2)))


def test_apply_program_inverse_with_conjugated_tensors_is_the_adjoint():
    program, tensors = compile_program(_gates(), 1, 1)
    x = _image(np.random.default_rng(5), (3, 2, 2))
    out = apply_program(program, tensors, x)
    back = apply_program(program, [jnp.conj(t) for t in tensors], out, inverse=True)
    np.testing.assert_allclose(back, x, atol=1e-12)


def test_slices_are_a_distinct_code_computing_the_same_thing():
    """The opt-in arithmetic of the one-qubit gates agrees with the default to
    rounding, in both directions and both precisions, at tensors with no symmetry."""
    rng = np.random.default_rng(6)
    gates = _gates() + [Gate(kind="CRY", qubits=(2, 1), tensor=HADAMARD, phase=0.0)]
    program, tensors = compile_program(gates, 1, 1)
    tensors = [t * jnp.asarray(1 + 0.2 * rng.standard_normal(t.shape)) for t in tensors]
    assert CircuitCode(program, slices=True) != CircuitCode(program)
    x = _image(rng, (4, 2, 2))
    for inverse in (False, True):
        default = apply_program(program, tensors, x, inverse=inverse)
        sliced = apply_program(program, tensors, x, inverse=inverse, slices=True)
        np.testing.assert_allclose(sliced, default, rtol=1e-13, atol=1e-13)
        single = apply_program(
            program, tensors, x.astype(jnp.complex64), inverse=inverse, slices=True
        )
        assert single.dtype == jnp.complex64
        np.testing.assert_allclose(single, default, rtol=1e-5, atol=1e-5)
