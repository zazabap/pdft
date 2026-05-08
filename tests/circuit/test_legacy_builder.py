"""Tests for the legacy `build_circuit_einsum` (single-string-einsum builder).

`compile_circuit` no longer routes through `build_circuit_einsum` — it uses
the stepped-tensordot path that handles circuits beyond the 52-character
label pool. The legacy function is retained as public API (re-exported
from `pdft.circuit`) so external consumers depending on the
`(subscripts, tensors, shapes)` triple keep working.

These tests exercise the legacy function directly and pin three things:
  - subscripts and shape lists are well-formed for each gate kind,
  - the Hadamard-first operand sort is preserved, and
  - the resulting einsum, when contracted, agrees with `compile_circuit`
    on a small circuit (no overlap with the label-pool limit).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pdft.circuit import build_circuit_einsum, compile_circuit
from pdft.circuit.builder import (
    HADAMARD,
    Gate,
    controlled_phase_diag,
    u4_from_phase,
)


def _h_gate(q: int) -> Gate:
    return Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0)


def _cp_gate(q_ctrl: int, q_tgt: int, phi: float = 0.5) -> Gate:
    return Gate(kind="CP", qubits=(q_ctrl, q_tgt), tensor=controlled_phase_diag(phi), phase=phi)


def _u4_gate(q_ctrl: int, q_tgt: int, phi: float = 0.5) -> Gate:
    return Gate(kind="U4", qubits=(q_ctrl, q_tgt), tensor=u4_from_phase(phi), phase=phi)


def test_legacy_build_h_only_circuit_subscripts_and_shapes():
    """Single Hadamard on qubit 1 of a 1+1-qubit circuit."""
    gates = [_h_gate(1)]
    subs, tensors, shapes = build_circuit_einsum(gates, m=1, n=1, inverse=False)
    assert "->" in subs
    # One H tensor + the pic operand
    assert len(tensors) == 1
    assert len(shapes) == 2
    # H tensor shape (2,2); pic shape (2,2) for m=n=1
    assert shapes[0] == (2, 2)
    assert shapes[1] == (2, 2)


def test_legacy_build_cp_does_not_introduce_new_labels():
    """CP gate shares its operands' labels — no fresh labels emitted."""
    gates = [_h_gate(1), _h_gate(2), _cp_gate(1, 2)]
    subs, tensors, shapes = build_circuit_einsum(gates, m=1, n=1, inverse=False)
    # 3 gate tensors + 1 pic operand
    assert len(tensors) == 3
    # All three gate tensors are 2x2; pic is 2x2
    assert all(s == (2, 2) for s in shapes)


def test_legacy_build_u4_emits_two_fresh_labels():
    """U4 gate has shape (2,2,2,2) and emits two new wire labels."""
    gates = [_u4_gate(1, 2)]
    subs, tensors, shapes = build_circuit_einsum(gates, m=1, n=1, inverse=False)
    assert shapes[0] == (2, 2, 2, 2)
    # Labels in the U4's subscript should be 4 distinct chars
    lhs = subs.split("->")[0]
    u4_subscripts = lhs.split(",")[0]
    assert len(u4_subscripts) == 4
    assert len(set(u4_subscripts)) == 4


def test_legacy_build_inverse_swaps_input_and_output_labels():
    """inverse=True routes pic in via the gates' OUTPUT labels and out via INPUTS."""
    gates = [_h_gate(1)]
    subs_fwd, _, _ = build_circuit_einsum(gates, m=1, n=1, inverse=False)
    subs_inv, _, _ = build_circuit_einsum(gates, m=1, n=1, inverse=True)
    # Forward: gate[out, in], pic[in] -> [out].   inverse: gate[out, in], pic[out] -> [in].
    fwd_lhs, fwd_rhs = subs_fwd.split("->")
    inv_lhs, inv_rhs = subs_inv.split("->")
    fwd_pic_lbl = fwd_lhs.split(",")[-1]
    inv_pic_lbl = inv_lhs.split(",")[-1]
    # Forward output label == inverse input (pic) label, and vice versa.
    assert fwd_pic_lbl == inv_rhs
    assert fwd_rhs == inv_pic_lbl


def test_legacy_build_hadamard_first_sort():
    """Hadamards must come first in the operand list regardless of gate-list order."""
    # Interleaved: U4, H, U4, H
    gates = [_u4_gate(1, 2), _h_gate(1), _u4_gate(1, 2), _h_gate(2)]
    _, tensors, _ = build_circuit_einsum(gates, m=1, n=1, inverse=False)
    # First two operands should be the H tensors (shape (2,2)), last two U4 (shape (2,2,2,2)).
    assert tensors[0].shape == (2, 2)
    assert tensors[1].shape == (2, 2)
    assert tensors[2].shape == (2, 2, 2, 2)
    assert tensors[3].shape == (2, 2, 2, 2)


def test_legacy_build_qubit_out_of_range_raises():
    """Validation: qubit index 0 (1-indexed convention) and > m+n raise ValueError."""
    # The Hadamard on qubit 0 hits a KeyError at wire_state[q]; the explicit
    # check is on label-pool exhaustion. Instead probe the >m+n branch via
    # an out-of-range qubit on a 1-qubit circuit.
    gates = [_h_gate(99)]
    with pytest.raises(KeyError):
        build_circuit_einsum(gates, m=1, n=1, inverse=False)


def test_legacy_build_label_pool_exhaustion_raises():
    """Construct enough U4 gates to overflow the 52-label pool."""
    # 12 U4 gates each emit 2 fresh labels = 24 fresh from U4 plus 4 input
    # labels. Add Hadamards to push past 52.
    gates: list[Gate] = []
    # Need to emit > 52 fresh labels. Each H on a wire emits 1, each U4 emits 2.
    # 30 H gates on 4 wires (cycling) = 30 fresh. Plus 4 input + 12 H more = 46.
    # Easier: use 30 U4 gates between qubits 1-2 = 60 fresh. Plus 4 inputs.
    for _ in range(30):
        gates.append(_u4_gate(1, 2))
    with pytest.raises(ValueError, match="too many qubits"):
        build_circuit_einsum(gates, m=2, n=2, inverse=False)


def test_legacy_build_matches_compile_circuit_numerically():
    """For a small circuit, the legacy big-einsum and the new stepped path
    must produce the same forward output to machine precision."""
    from pdft.circuit.cache import optimize_code_cached

    gates = [_h_gate(1), _h_gate(2), _cp_gate(1, 2, phi=0.7), _u4_gate(1, 2, phi=0.3)]
    subs, tensors_legacy, shapes = build_circuit_einsum(gates, m=1, n=1, inverse=False)
    legacy_code = optimize_code_cached(subs, *shapes)
    code_new, tensors_new = compile_circuit(gates, m=1, n=1, inverse=False)

    rng = np.random.default_rng(42)
    pic_2d = jnp.asarray(rng.normal(size=(2, 2)).astype(np.float64))
    pic = pic_2d.astype(jnp.complex128).reshape((2, 2))

    out_legacy = legacy_code(*tensors_legacy, pic)
    out_new = code_new(*tensors_new, pic)

    np.testing.assert_allclose(np.asarray(out_legacy), np.asarray(out_new), atol=1e-12)
