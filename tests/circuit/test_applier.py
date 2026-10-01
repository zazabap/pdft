"""The applier: one gate at a time through `CircuitCode` and `apply_program`, checked against an einsum."""

from __future__ import annotations

import string

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.circuit.qft import qft_gates_1d
from pdft.circuit.builder import (
    GATE_SHAPES,
    HADAMARD,
    CircuitCode,
    Gate,
    Program,
    _walk,
    apply_circuit,
    apply_program,
    compile_program,
)

from ..helpers import (
    BASES,
    case_rng,
    complex_image,
    complex_normal,
    generic,
    random_unitary,
    small_circuit,
)


def test_apply_program_is_apply_circuit_on_one_double_precision_image():
    program, tensors = compile_program(small_circuit(), 1, 1)
    x = complex_image((2, 2), seed=2)
    code = CircuitCode(program)
    assert jnp.array_equal(
        apply_program(program, tensors, x), apply_circuit(tensors, code, 1, 1, x)
    )


def test_apply_program_carries_leading_axes():
    gates = qft_gates_1d(2, 0) + qft_gates_1d(3, 2)
    program, tensors = compile_program(gates, 2, 3)
    stack = complex_image((2, 5, 4, 8), seed=3)
    out = apply_program(program, tensors, stack)
    assert out.shape == stack.shape
    for i in range(2):
        for j in range(5):
            assert jnp.array_equal(out[i, j], apply_program(program, tensors, stack[i, j]))


def test_apply_program_follows_the_precision_of_the_image():
    program, tensors = compile_program(small_circuit(), 1, 1)
    x = jnp.asarray(np.random.default_rng(4).random((2, 2)))
    double = apply_program(program, tensors, x)
    single = apply_program(program, tensors, x.astype(jnp.float32))
    assert double.dtype == jnp.complex128 and single.dtype == jnp.complex64
    assert apply_program(program, tensors, x.astype(jnp.complex64)).dtype == jnp.complex64
    np.testing.assert_allclose(single, double, atol=1e-6)


def test_apply_program_refuses_another_image_size():
    program, tensors = compile_program(small_circuit(), 1, 1)
    with pytest.raises(ValueError, match="image shape"):
        apply_program(program, tensors, jnp.zeros((4, 2)))


def test_apply_program_inverse_with_conjugated_tensors_is_the_adjoint():
    program, tensors = compile_program(small_circuit(), 1, 1)
    x = complex_image((3, 2, 2), seed=5)
    out = apply_program(program, tensors, x)
    back = apply_program(program, [jnp.conj(t) for t in tensors], out, inverse=True)
    np.testing.assert_allclose(back, x, atol=1e-12)


def test_slices_are_a_distinct_code_computing_the_same_thing():
    """The opt-in arithmetic of the one-qubit gates agrees with the default to
    rounding, in both directions and both precisions, at tensors with no symmetry."""
    rng = np.random.default_rng(6)
    gates = small_circuit() + [Gate(kind="CRY", qubits=(2, 1), tensor=HADAMARD, phase=0.0)]
    program, tensors = compile_program(gates, 1, 1)
    tensors = [t * jnp.asarray(1 + 0.2 * rng.standard_normal(t.shape)) for t in tensors]
    assert CircuitCode(program, slices=True) != CircuitCode(program)
    x = jnp.asarray(complex_normal(rng, (4, 2, 2)))
    for inverse in (False, True):
        default = apply_program(program, tensors, x, inverse=inverse)
        sliced = apply_program(program, tensors, x, inverse=inverse, slices=True)
        np.testing.assert_allclose(sliced, default, rtol=1e-13, atol=1e-13)
        single = apply_program(
            program, tensors, x.astype(jnp.complex64), inverse=inverse, slices=True
        )
        assert single.dtype == jnp.complex64
        np.testing.assert_allclose(single, default, rtol=1e-5, atol=1e-5)


def test_a_gate_outside_the_registers_or_of_an_unknown_kind_is_refused():
    pic = jnp.zeros((2, 2), dtype=jnp.complex128)
    stray = Program(1, 1, (("H", (3,)),), (0,))
    with pytest.raises(ValueError, match=r"qubit index 3 out of range \(1..2\)"):
        CircuitCode(stray)(HADAMARD, pic)
    unknown = Program(1, 1, (("SWAP", (1, 2)),), (0,))
    with pytest.raises(AssertionError, match="unknown gate kind: SWAP"):
        CircuitCode(unknown)(HADAMARD, pic)


def _as_one_einsum(steps, tensors, m, n, pic, inverse):
    """The circuit read as a single einsum: a second, independent statement of what a gate list means.

    Every wire carries a label. A one-qubit or dense gate reads its wires'
    labels and writes new ones; a diagonal gate only reads them. The image
    enters on the first labels and leaves on the last, least significant qubit
    last within each register; the other way round for the inverse, which is
    the transpose. ``tensors`` are in the order the gates act.
    """
    labels = iter(string.ascii_letters)
    wire = {q: next(labels) for q in range(1, m + n + 1)}
    entering = dict(wire)
    subscripts = []
    for kind, qubits in steps:
        read = [wire[q] for q in qubits]
        if kind == "CP":
            subscripts.append("".join(read))
        else:
            written = [next(labels) for _ in qubits]
            subscripts.append("".join(written + read))
            wire.update(zip(qubits, written))
    order = [*range(m, 0, -1), *range(m + n, m, -1)]
    source, target = (wire, entering) if inverse else (entering, wire)
    image_in = "".join(source[q] for q in order)
    image_out = "".join(target[q] for q in order)
    # "greedy": the optimal path search is exponential in the number of tensors
    return jnp.einsum(
        ",".join([*subscripts, image_in]) + "->" + image_out, *tensors, pic, optimize="greedy"
    )


# every registered circuit the einsum can express: it has no controlled-rotation kind
EINSUM_CASES = [
    case
    for case, make in BASES.items()
    if hasattr(make(), "program") and "CRY" not in {kind for kind, _ in make().program.steps}
]


@pytest.mark.parametrize("case", EINSUM_CASES)
@pytest.mark.parametrize("inverse", [False, True])
def test_the_walk_agrees_with_one_einsum(case, inverse):
    """At random tensors, so a gate applied transposed or on the wrong wire shows: both
    readings are multilinear in the tensors, and neither needs them unitary."""
    program = BASES[case]().program
    rng = case_rng(case)
    stored = [
        jnp.asarray(complex_normal(rng, GATE_SHAPES[kind])) for kind, _ in program.sorted_steps
    ]
    pic = jnp.asarray(complex_normal(rng, (2,) * (program.m + program.n)))
    walked = CircuitCode(program, inverse=inverse)(*stored, pic)
    in_order = [stored[slot] for slot in program.slot]
    reference = _as_one_einsum(program.steps, in_order, program.m, program.n, pic, inverse)
    np.testing.assert_allclose(walked, reference, rtol=1e-12, atol=1e-12)


def test_inverse_walks_the_steps_backwards_with_swapped_legs():
    """With conjugated tensors the inverse code is the adjoint, so for unitary gates it
    undoes the forward code. The gates here have no symmetry, so both halves are needed:
    with the steps in forward order, or with the legs not swapped, the round trip fails."""
    rng = np.random.default_rng(1)
    program, _ = compile_program(small_circuit(), 1, 1)
    tensors = [
        jnp.asarray(np.exp(1j * rng.uniform(-3, 3, (2, 2))))
        if kind == "CP"
        else jnp.asarray(random_unitary(rng, 4 if kind == "U4" else 2)).reshape(GATE_SHAPES[kind])
        for kind, _ in program.sorted_steps
    ]
    assert not any(
        jnp.allclose(t, jnp.swapaxes(t.reshape(t.shape[0], -1), 0, 1).reshape(t.shape))
        for t in tensors
        if t.ndim == 2
    )
    pic = complex_image((2, 2), seed=1)
    out = CircuitCode(program)(*tensors, pic)
    adjoint = [jnp.conj(t) for t in tensors]
    np.testing.assert_allclose(CircuitCode(program, inverse=True)(*adjoint, out), pic, atol=1e-12)
    # neither the forward walk of the adjoint tensors nor the inverse walk of the plain ones
    assert not jnp.allclose(CircuitCode(program)(*adjoint, out), pic, atol=1e-6)
    assert not jnp.allclose(CircuitCode(program, inverse=True)(*tensors, out), pic, atol=1e-6)


def test_a_code_called_directly_reads_the_qubits_from_the_first_axes():
    """``basis.code(*tensors, pic)`` takes the first ``m + n`` axes as the qubits. An axis
    after them rides along, and an array with too few is refused, not misread."""
    basis = generic(pdft.RichBasis(m=1, n=2), np.random.default_rng(4))
    stack = complex_image((2, 2, 2, 3), seed=4)
    out = basis.code(*basis.tensors, stack)
    assert out.shape == stack.shape
    for k in range(3):
        np.testing.assert_allclose(
            out[..., k], basis.code(*basis.tensors, stack[..., k]), rtol=0, atol=1e-14
        )
    with pytest.raises((ValueError, IndexError)):
        basis.code(*basis.tensors, complex_image((2, 4)))


def _primitives(jaxpr) -> set[str]:
    """The names of the primitives in a traced graph, those inside jitted calls included."""
    names = set()
    for equation in jaxpr.eqns:
        names.add(equation.primitive.name)
        for value in equation.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                names |= _primitives(inner)
    return names


@pytest.mark.parametrize("kind", ["H", "CRY"])
def test_the_slice_arithmetic_is_what_runs_when_asked_for(kind):
    """The two arithmetics agree to rounding, so agreement cannot show which one ran. With
    slices a one-qubit gate, and the block of a controlled rotation, involve no
    contraction at all; and that holds for each way of asking: the walk, a
    ``CircuitCode``, ``apply_program``."""
    steps = (("H", (1,)), ("H", (2,))) if kind == "H" else (("CRY", (1, 2)),)
    program = Program(1, 1, steps, tuple(range(len(steps))))
    tensors = [HADAMARD] * len(steps)
    pic = complex_image((2, 2), seed=2)
    ways = {
        "walk": lambda flag: lambda *ts: _walk(program, False, flag, ts, pic),
        "code": lambda flag: lambda *ts: CircuitCode(program, slices=flag)(*ts, pic),
        "apply_program": lambda flag: lambda *ts: apply_program(program, ts, pic, slices=flag),
    }
    for name, way in ways.items():
        contracts = {
            flag: "dot_general" in _primitives(jax.make_jaxpr(way(flag))(*tensors).jaxpr)
            for flag in (False, True)
        }
        assert contracts == {False: True, True: False}, name
