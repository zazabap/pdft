"""The gate constructors of `pdft.circuit.builder`, the tensors they make and reading phases back."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pdft.bases.circuit.qft import qft_gates, qft_gates_1d
from pdft.circuit.builder import (
    HADAMARD,
    CircuitCode,
    Program,
    check_qubits,
    controlled_phase_diag,
    cp_gate,
    extract_phases,
    hadamard_gate,
    hadamards_then_layers,
    phase_gate,
    two_registers,
    u4_from_phase,
    u4_gate,
)

from ..helpers import complex_image, gate_structure


def test_hadamard_gate():
    g = hadamard_gate(3)
    assert (g["kind"], g["qubits"], g["phase"]) == ("H", (3,), 0.0) and g["tensor"] is HADAMARD


def test_phase_gate_has_two_forms_of_one_operator():
    cp, u4 = cp_gate(2, 1, 0.7), u4_gate(2, 1, 0.7)
    assert (cp["kind"], u4["kind"]) == ("CP", "U4") and cp["qubits"] == u4["qubits"] == (2, 1)
    assert cp["phase"] == u4["phase"] == 0.7
    assert jnp.array_equal(cp["tensor"], controlled_phase_diag(0.7))
    assert jnp.array_equal(u4["tensor"], u4_from_phase(0.7))
    # the dense tensor is the 4x4 gate whose diagonal the compact one lists
    assert jnp.array_equal(jnp.diag(u4["tensor"].reshape(4, 4)), cp["tensor"].reshape(-1))
    with pytest.raises(ValueError, match="parametrization must be 'cp' or 'u4'"):
        phase_gate("o4")


def test_controlled_puts_the_block_where_the_control_is_one():
    from pdft.circuit.builder import controlled

    block = complex_image((2, 2))
    # rows (out_c, out_t), columns (in_c, in_t)
    matrix = np.asarray(controlled(block)).reshape(4, 4)
    expected = np.zeros((4, 4), dtype=complex)
    expected[:2, :2] = np.eye(2)
    expected[2:, 2:] = np.asarray(block)
    np.testing.assert_array_equal(matrix, expected)
    with pytest.raises(ValueError, match=r"the block of a controlled gate is \(2, 2\)"):
        controlled(jnp.ones(2))
    # the dense controlled phase is that, with a phase for the block
    np.testing.assert_array_equal(
        np.asarray(u4_from_phase(0.7)).reshape(4, 4), np.diag([1, 1, 1, np.exp(0.7j)])
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_identity_tensors_make_every_gate_kind_do_nothing(inverse):
    from pdft.circuit.builder import GATE_KINDS, GATE_SHAPES, identity_tensor

    assert tuple(GATE_SHAPES) == GATE_KINDS
    steps = (("H", (1,)), ("CP", (1, 3)), ("U4", (3, 2)), ("CRY", (2, 4)), ("H", (4,)))
    program = Program(2, 2, steps, tuple(range(len(steps))))
    tensors = [identity_tensor(kind) for kind, _ in steps]
    assert [t.shape for t in tensors] == [GATE_SHAPES[kind] for kind, _ in steps]
    pic = complex_image((2, 2, 2, 2), seed=1)
    # to rounding, not to the bit: a GPU's contraction with an identity is not exact
    np.testing.assert_allclose(
        CircuitCode(program, inverse=inverse)(*tensors, pic), pic, rtol=0, atol=1e-14
    )
    with pytest.raises(AssertionError, match="unknown gate kind: SWAP"):
        identity_tensor("SWAP")


def test_two_registers_puts_columns_after_rows():
    assert gate_structure(qft_gates(2, 3)) == gate_structure(two_registers(qft_gates_1d, 2, 3))
    assert gate_structure(qft_gates(2, 1)) == [
        ("H", (1,)),
        ("CP", (2, 1)),
        ("H", (2,)),
        ("H", (3,)),
    ]
    for m, n in ((0, 2), (2, 0)):
        with pytest.raises(ValueError, match="must be >= 1"):
            two_registers(qft_gates_1d, m, n)
        with pytest.raises(ValueError, match="must be >= 1"):
            check_qubits(m, n)


def test_hadamards_then_layers_splits_the_phases_between_the_registers():
    calls = []

    def layer(n_qubits, offset, phases, gate):
        calls.append((n_qubits, offset, list(phases)))
        return [gate(offset + 1, offset + 2, phi) for phi in phases]

    def count(n_qubits):
        return n_qubits - 1

    gates, n_row, n_col = hadamards_then_layers(layer, count, 3, 2, [0.1, 0.2, 0.3], "u4")
    assert (n_row, n_col) == (2, 1)
    assert calls == [(3, 0, [0.1, 0.2]), (2, 3, [0.3])]
    hadamards = [("H", (q,)) for q in range(1, 6)]
    assert gate_structure(gates) == hadamards + [("U4", (1, 2))] * 2 + [("U4", (4, 5))]
    assert [g["phase"] for g in gates[5:]] == [0.1, 0.2, 0.3]

    zeros, _, _ = hadamards_then_layers(layer, count, 3, 2, None, "cp")
    assert gate_structure(zeros) == hadamards + [("CP", (1, 2))] * 2 + [("CP", (4, 5))]
    assert [g["phase"] for g in zeros[5:]] == [0.0] * 3

    with pytest.raises(ValueError, match=r"length 3 \(2 row \+ 1 column gates\), got 2"):
        hadamards_then_layers(layer, count, 3, 2, [0.1, 0.2], "cp")
    with pytest.raises(ValueError, match="parametrization must be 'cp' or 'u4'"):
        hadamards_then_layers(layer, count, 3, 2, None, "dense")
    with pytest.raises(ValueError, match="must be >= 1"):
        hadamards_then_layers(layer, count, 0, 2, None, "cp")


def test_phase_list_defaults_to_zeros_and_checks_the_length():
    from pdft.circuit.builder import phase_list

    assert phase_list(None, 3, "three phases") == [0.0, 0.0, 0.0]
    assert phase_list(range(3), 3, "three phases") == [0.0, 1.0, 2.0]
    assert all(isinstance(p, float) for p in phase_list(np.arange(2), 2, "two"))
    with pytest.raises(ValueError, match="^three phases, got 2$"):
        phase_list([1, 2], 3, "three phases")


def test_extract_phases_reads_the_listed_tensors_in_order():
    tensors = [HADAMARD, controlled_phase_diag(0.3), controlled_phase_diag(-1.1)]
    assert extract_phases(tensors, [2, 1]) == pytest.approx([-1.1, 0.3])


def test_select_last_n_cp_indices_returns_what_there_is():
    from pdft.circuit.builder import select_last_n_cp_indices

    tensors = [HADAMARD, controlled_phase_diag(0.1), HADAMARD, controlled_phase_diag(0.2)]
    assert select_last_n_cp_indices(tensors, 1) == [3]
    assert select_last_n_cp_indices(tensors, 2) == [1, 3]
    assert select_last_n_cp_indices(tensors, 5) == [1, 3]
