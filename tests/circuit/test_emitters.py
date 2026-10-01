"""The gate constructors and the one QFT skeleton every QFT-topology basis shares."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pdft  # noqa: F401  (enables x64)
from pdft.bases.circuit.dct4 import dct4_ft_mat, dct4_ift_mat
from pdft.bases.circuit.mera import mera_gates
from pdft.bases.circuit.qft import _qft_gates_1d, ft_mat, ift_mat, qft_gates, qft_gates_1d
from pdft.bases.circuit.real_rich import _real_rich_qft_gates_1d
from pdft.bases.circuit.rich import _rich_qft_gates_1d
from pdft.bases.circuit.tebd import tebd_gates
from pdft.circuit.builder import (
    HADAMARD,
    apply_circuit,
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


def _structure(gates):
    return [(g["kind"], g["qubits"]) for g in gates]


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


def test_qft_skeleton_is_shared_by_the_three_qft_topology_bases():
    plain = _qft_gates_1d(3, 2)
    assert _qft_gates_1d is qft_gates_1d
    assert _structure(plain) == [
        ("H", (3,)),
        ("CP", (4, 3)),
        ("CP", (5, 3)),
        ("H", (4,)),
        ("CP", (5, 4)),
        ("H", (5,)),
    ]
    assert [g["phase"] for g in plain if g["kind"] == "CP"] == [np.pi / 2, np.pi / 4, np.pi / 2]
    dense = [(kind.replace("CP", "U4"), qubits) for kind, qubits in _structure(plain)]
    assert _structure(_rich_qft_gates_1d(3, 2)) == dense
    assert _structure(_real_rich_qft_gates_1d(3, 2)) == dense
    # rich starts at the QFT operator, real-rich at the identity
    for rich, real, cp in zip(_rich_qft_gates_1d(3, 2), _real_rich_qft_gates_1d(3, 2), plain):
        if cp["kind"] == "CP":
            assert jnp.array_equal(rich["tensor"], u4_from_phase(cp["phase"]))
            assert (
                jnp.array_equal(real["tensor"].reshape(4, 4), jnp.eye(4)) and real["phase"] == 0.0
            )


def test_two_registers_puts_columns_after_rows():
    assert _structure(qft_gates(2, 3)) == _structure(two_registers(qft_gates_1d, 2, 3))
    assert _structure(qft_gates(2, 1)) == [("H", (1,)), ("CP", (2, 1)), ("H", (2,)), ("H", (3,))]
    for m, n in ((0, 2), (2, 0)):
        with pytest.raises(ValueError, match="must be >= 1"):
            two_registers(qft_gates_1d, m, n)
        with pytest.raises(ValueError, match="must be >= 1"):
            check_qubits(m, n)


def test_extract_phases_reads_the_listed_tensors_in_order():
    tensors = [HADAMARD, controlled_phase_diag(0.3), controlled_phase_diag(-1.1)]
    assert extract_phases(tensors, [2, 1]) == pytest.approx([-1.1, 0.3])


def test_family_phase_helpers_are_the_one_implementation():
    from pdft.bases.circuit import entangled_qft, mera, tebd
    from pdft.circuit.builder import select_last_n_cp_indices

    for module, indices, phases in (
        (tebd, "get_tebd_gate_indices", "extract_tebd_phases"),
        (mera, "get_mera_gate_indices", "extract_mera_phases"),
    ):
        assert getattr(module, indices) is select_last_n_cp_indices
        assert getattr(module, phases) is extract_phases

    # every helper keeps the parameter names it has upstream, for keyword callers
    tensors = [HADAMARD, controlled_phase_diag(0.3), HADAMARD, controlled_phase_diag(-1.1)]
    for module, indices, count, phases, which in (
        (
            entangled_qft,
            "get_entangle_tensor_indices",
            "n_entangle",
            "extract_entangle_phases",
            "entangle_indices",
        ),
        (tebd, "get_tebd_gate_indices", "n_gates", "extract_tebd_phases", "gate_indices"),
        (mera, "get_mera_gate_indices", "n_gates", "extract_mera_phases", "gate_indices"),
    ):
        found = getattr(module, indices)(tensors=tensors, **{count: 2})
        assert found == select_last_n_cp_indices(tensors, 2) == [1, 3]
        read = getattr(module, phases)(tensors=tensors, **{which: found})
        assert read == extract_phases(tensors, found) == pytest.approx([0.3, -1.1])


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
    assert _structure(gates) == hadamards + [("U4", (1, 2))] * 2 + [("U4", (4, 5))]
    assert [g["phase"] for g in gates[5:]] == [0.1, 0.2, 0.3]

    zeros, _, _ = hadamards_then_layers(layer, count, 3, 2, None, "cp")
    assert _structure(zeros) == hadamards + [("CP", (1, 2))] * 2 + [("CP", (4, 5))]
    assert [g["phase"] for g in zeros[5:]] == [0.0] * 3

    with pytest.raises(ValueError, match=r"length 3 \(2 row \+ 1 column gates\), got 2"):
        hadamards_then_layers(layer, count, 3, 2, [0.1, 0.2], "cp")
    with pytest.raises(ValueError, match="parametrization must be 'cp' or 'u4'"):
        hadamards_then_layers(layer, count, 3, 2, None, "dense")
    with pytest.raises(ValueError, match="must be >= 1"):
        hadamards_then_layers(layer, count, 0, 2, None, "cp")


def test_tebd_emits_a_ring_per_register():
    gates, n_row, n_col = tebd_gates(3, 2, phases=[1, 2, 3, 4, 5])
    assert (n_row, n_col) == (3, 2)
    assert _structure(gates[:5]) == [("H", (q,)) for q in range(1, 6)]
    # (i, i+1) along the register, then the wrap-around back to its first qubit
    assert [g["qubits"] for g in gates[5:]] == [(1, 2), (2, 3), (3, 1), (4, 5), (5, 4)]
    assert [g["phase"] for g in gates[5:]] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert {g["kind"] for g in tebd_gates(2, 2, parametrization="u4")[0][4:]} == {"U4"}


def test_mera_emits_disentanglers_then_isometries_per_layer():
    gates, n_row, n_col = mera_gates(4, 1, phases=range(6))
    assert (n_row, n_col) == (6, 0)
    assert [g["qubits"] for g in gates[5:]] == [(2, 3), (4, 1), (1, 2), (3, 4), (2, 4), (1, 3)]
    assert [g["phase"] for g in gates[5:]] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # a one-qubit register gets no layer; the other one starts after it
    gates, n_row, n_col = mera_gates(1, 2)
    assert (n_row, n_col) == (0, 2) and [g["qubits"] for g in gates[3:]] == [(3, 2), (2, 3)]
    with pytest.raises(ValueError, match="m must be a power of 2 when >= 2, got m=3"):
        mera_gates(3, 2)
    with pytest.raises(ValueError, match="n must be a power of 2 when >= 2, got n=6"):
        mera_gates(2, 6)


def test_the_julia_transform_names_are_one_function():
    assert ft_mat is ift_mat is apply_circuit
    assert dct4_ft_mat is dct4_ift_mat is apply_circuit


def test_controlled_puts_the_block_where_the_control_is_one():
    from pdft.circuit.builder import controlled

    rng = np.random.default_rng(0)
    block = jnp.asarray(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    matrix = np.asarray(controlled(block)).reshape(
        4, 4
    )  # rows (out_c, out_t), columns (in_c, in_t)
    expected = np.zeros((4, 4), dtype=complex)
    expected[:2, :2] = np.eye(2)
    expected[2:, 2:] = np.asarray(block)
    np.testing.assert_array_equal(matrix, expected)
    # the three gates built from it
    np.testing.assert_array_equal(
        np.asarray(u4_from_phase(0.7)).reshape(4, 4), np.diag([1, 1, 1, np.exp(0.7j)])
    )
    from pdft.bases.circuit.dct4 import _cnot_u4, _cry_u4, _ry

    np.testing.assert_array_equal(
        np.asarray(_cnot_u4()).reshape(4, 4),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    )
    np.testing.assert_array_equal(np.asarray(_cry_u4(0.4))[1, :, 1, :], np.asarray(_ry(0.4)))


@pytest.mark.parametrize("inverse", [False, True])
def test_identity_tensors_make_every_gate_kind_do_nothing(inverse):
    from pdft.circuit.builder import GATE_KINDS, GATE_SHAPES, CircuitCode, Program, identity_tensor

    assert tuple(GATE_SHAPES) == GATE_KINDS
    steps = (("H", (1,)), ("CP", (1, 3)), ("U4", (3, 2)), ("CRY", (2, 4)), ("H", (4,)))
    program = Program(2, 2, steps, tuple(range(len(steps))))
    tensors = [identity_tensor(kind) for kind, _ in steps]
    assert [t.shape for t in tensors] == [GATE_SHAPES[kind] for kind, _ in steps]
    rng = np.random.default_rng(1)
    pic = jnp.asarray(rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2)))
    # to rounding, not to the bit: a GPU's contraction with an identity is not exact
    np.testing.assert_allclose(
        CircuitCode(program, inverse=inverse)(*tensors, pic), pic, rtol=0, atol=1e-14
    )
    with pytest.raises(AssertionError, match="unknown gate kind: SWAP"):
        identity_tensor("SWAP")


def test_phase_list_defaults_to_zeros_and_checks_the_length():
    from pdft.circuit.builder import phase_list

    assert phase_list(None, 3, "three phases") == [0.0, 0.0, 0.0]
    assert phase_list(range(3), 3, "three phases") == [0.0, 1.0, 2.0]
    assert all(isinstance(p, float) for p in phase_list(np.arange(2), 2, "two"))
    with pytest.raises(ValueError, match="^three phases, got 2$"):
        phase_list([1, 2], 3, "three phases")


def test_dct4_twiddles_record_their_angle_in_both_forms():
    from pdft.bases.circuit.dct4 import _dct4_gates_1d

    dense = _dct4_gates_1d(3, offset=0, parametrization="o4")
    blocks = _dct4_gates_1d(3, offset=0, parametrization="controlled")
    assert [g["qubits"] for g in dense] == [g["qubits"] for g in blocks]
    twiddles = [(a, b) for a, b in zip(dense, blocks) if b["kind"] == "CRY"]
    assert len(twiddles) == 3 and all(a["kind"] == "U4" for a, _ in twiddles)
    for a, b in twiddles:
        assert a["phase"] == b["phase"] > 0
        # the dense form is the block, controlled
        np.testing.assert_array_equal(np.asarray(a["tensor"])[1, :, 1, :], np.asarray(b["tensor"]))
    assert [g["kind"] for g in dense if g["kind"] != "U4"] == [
        g["kind"] for g in blocks if g["kind"] not in ("U4", "CRY")
    ]
