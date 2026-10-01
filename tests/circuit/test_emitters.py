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
        (entangled_qft, "get_entangle_tensor_indices", "extract_entangle_phases"),
        (tebd, "get_tebd_gate_indices", "extract_tebd_phases"),
        (mera, "get_mera_gate_indices", "extract_mera_phases"),
    ):
        assert getattr(module, indices) is select_last_n_cp_indices
        assert getattr(module, phases) is extract_phases


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
