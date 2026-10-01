"""The gate sequence each circuit family emits, and the upstream helper names that read it back."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases.circuit.entangled_qft import extract_entangle_phases, get_entangle_tensor_indices
from pdft.bases.circuit.mera import mera_gates
from pdft.bases.circuit.qft import _qft_gates_1d, qft_gates_1d
from pdft.bases.circuit.real_rich import _real_rich_qft_gates_1d
from pdft.bases.circuit.rich import _rich_qft_gates_1d
from pdft.bases.circuit.tebd import tebd_gates
from pdft.circuit.builder import (
    HADAMARD,
    controlled_phase_diag,
    u4_from_phase,
)

from ...helpers import gate_structure


def test_qft_skeleton_is_shared_by_the_three_qft_topology_bases():
    plain = _qft_gates_1d(3, 2)
    assert _qft_gates_1d is qft_gates_1d
    assert gate_structure(plain) == [
        ("H", (3,)),
        ("CP", (4, 3)),
        ("CP", (5, 3)),
        ("H", (4,)),
        ("CP", (5, 4)),
        ("H", (5,)),
    ]
    assert [g["phase"] for g in plain if g["kind"] == "CP"] == [np.pi / 2, np.pi / 4, np.pi / 2]
    dense = [(kind.replace("CP", "U4"), qubits) for kind, qubits in gate_structure(plain)]
    assert gate_structure(_rich_qft_gates_1d(3, 2)) == dense
    assert gate_structure(_real_rich_qft_gates_1d(3, 2)) == dense
    # rich starts at the QFT operator, real-rich at the identity
    for rich, real, cp in zip(_rich_qft_gates_1d(3, 2), _real_rich_qft_gates_1d(3, 2), plain):
        if cp["kind"] == "CP":
            assert jnp.array_equal(rich["tensor"], u4_from_phase(cp["phase"]))
            assert (
                jnp.array_equal(real["tensor"].reshape(4, 4), jnp.eye(4)) and real["phase"] == 0.0
            )


def test_tebd_emits_a_ring_per_register():
    gates, n_row, n_col = tebd_gates(3, 2, phases=[1, 2, 3, 4, 5])
    assert (n_row, n_col) == (3, 2)
    assert gate_structure(gates[:5]) == [("H", (q,)) for q in range(1, 6)]
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
    # so does the base rotation of each level, pi / (2 * size) for sizes 8, 4, 2
    rotations = [g["phase"] for g in dense if g["kind"] == "H" and g["phase"]]
    assert rotations == pytest.approx([np.pi / 16, np.pi / 8, np.pi / 4])


def test_front_entanglers_are_found_by_the_program_not_by_position():
    """The upstream helper takes the last compact-CP tensors, which are the entanglers
    only when they are emitted last. The program knows which gates couple the registers
    wherever they sit."""
    phases = [0.1, 0.4, 1.7]
    for position in ("back", "front"):
        basis = pdft.EntangledQFTBasis(m=3, n=3, entangle_phases=phases, entangle_position=position)
        coupling = basis.program.tensor_indices(kind="CP", register="both")
        assert len(coupling) == basis.n_entangle == 3
        np.testing.assert_allclose(
            extract_entangle_phases(basis.tensors, coupling), phases, atol=1e-15
        )
        by_position = get_entangle_tensor_indices(basis.tensors, basis.n_entangle)
        assert (by_position == coupling) == (position == "back")


def test_the_phase_helpers_keep_their_upstream_parameter_names():
    """Each family has upstream's two helpers under its own name, callable by keyword."""
    from pdft.bases.circuit import entangled_qft, mera, tebd

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
        assert found == [1, 3]
        assert getattr(module, phases)(tensors=tensors, **{which: found}) == pytest.approx(
            [0.3, -1.1]
        )


def test_controlled_phases_between_fixed_qubits_name_the_control_first():
    """The order of a gate's qubits is the order of its tensor's axes. A controlled phase
    starts symmetric, so the order shows only once training has made the two axes
    differ, and a checkpoint from before would then be read transposed."""
    entangled = pdft.EntangledQFTBasis(m=2, n=3).program
    coupling = entangled.tensor_indices(kind="CP", register="both")
    # row qubit first, then the column qubit it is paired with
    assert [entangled.sorted_steps[i][1] for i in coupling] == [(2, 5), (1, 4)]
    # the DCT-IV sign gate: the branch qubit first
    signs = [qubits for kind, qubits in pdft.DCT4Basis(m=3, n=1).program.steps if kind == "CP"]
    assert signs == [(2, 1), (3, 2)]
