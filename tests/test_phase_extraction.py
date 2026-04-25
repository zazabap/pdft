"""Tests + parity for phase-extraction helpers."""
from pathlib import Path

import numpy as np

from pdft.basis import EntangledQFTBasis, MERABasis, TEBDBasis
from pdft.entangled_qft import extract_entangle_phases, get_entangle_tensor_indices
from pdft.mera import extract_mera_phases, get_mera_gate_indices
from pdft.tebd import extract_tebd_phases, get_tebd_gate_indices

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


# ---------- EntangledQFT ----------


def test_entangle_phase_extraction_round_trip():
    phases = [0.1, 0.4, 1.7]
    b = EntangledQFTBasis(m=3, n=3, entangle_phases=phases)
    indices = get_entangle_tensor_indices(b.tensors, b.n_entangle)
    assert len(indices) == 3
    extracted = extract_entangle_phases(b.tensors, indices)
    np.testing.assert_allclose(extracted, phases, atol=1e-12)


def test_entangle_phase_extraction_matches_julia():
    g = np.load(GOLDENS / "phase_extraction_3x3.npz")
    input_phases = [float(x) for x in g["input_phases"]]
    n_ent = int(g["n_entangle"][0])

    b = EntangledQFTBasis(m=3, n=3, entangle_phases=input_phases)
    indices = get_entangle_tensor_indices(b.tensors, n_ent)
    extracted = extract_entangle_phases(b.tensors, indices)

    # Phases should match Julia bit-exactly (up to round-trip through angle())
    np.testing.assert_allclose(
        extracted, [float(x) for x in g["extracted_phases"]], atol=1e-12
    )


# ---------- TEBD ----------


def test_tebd_phase_extraction_round_trip():
    phases = [0.2 * i for i in range(4)]  # m + n = 4 gates for (2, 2)
    b = TEBDBasis(m=2, n=2, phases=phases)
    n_gates = b.n_row_gates + b.n_col_gates
    indices = get_tebd_gate_indices(b.tensors, n_gates)
    assert len(indices) == n_gates
    extracted = extract_tebd_phases(b.tensors, indices)
    np.testing.assert_allclose(extracted, phases, atol=1e-12)


# ---------- MERA ----------


def test_mera_phase_extraction_round_trip():
    phases = [0.1 * (i + 1) for i in range(4)]  # 2 + 2 = 4 gates for (2, 2)
    b = MERABasis(m=2, n=2, phases=phases)
    n_gates = b.n_row_gates + b.n_col_gates
    indices = get_mera_gate_indices(b.tensors, n_gates)
    assert len(indices) == n_gates
    extracted = extract_mera_phases(b.tensors, indices)
    np.testing.assert_allclose(extracted, phases, atol=1e-12)
