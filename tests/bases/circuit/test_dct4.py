import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import dct

from pdft.bases.base import DCT4Basis
from pdft.bases.circuit.dct4 import (
    _cnot_u4,
    _cry_u4,
    _dct4_gates_1d,
    _ry,
    dct4_code,
    dct4_ft_mat,
    dct4_ift_mat,
)


def _bitrev_perm(n_qubits: int) -> np.ndarray:
    """Bit-reversal permutation matrix on 2**n_qubits indices."""
    size = 2**n_qubits
    P = np.zeros((size, size))
    for i in range(size):
        r = int(format(i, f"0{n_qubits}b")[::-1], 2)
        P[r, i] = 1.0
    return P


def _dct4_op(n_qubits: int) -> np.ndarray:
    """Expected 1D forward operator: bit-reversed orthonormal DCT-IV."""
    size = 2**n_qubits
    D4 = dct(np.eye(size), type=4, norm="ortho", axis=0)
    return _bitrev_perm(n_qubits) @ D4


# ---------- gate tensors ----------


def test_ry_is_real_orthogonal():
    R = np.asarray(_ry(0.7))
    assert np.allclose(R.imag, 0.0, atol=1e-15)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)


def test_cnot_u4_structure():
    M = np.asarray(_cnot_u4()).reshape(4, 4)  # [oc ot, ic it]
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(M, expected, atol=1e-12)


def test_cry_u4_control_blocks():
    theta = 0.9
    M = np.asarray(_cry_u4(theta)).reshape(4, 4)  # rows (oc,ot), cols (ic,it)
    # control preserved: block-diag(I on target | R_y(theta) on target)
    Ry = np.asarray(_ry(theta))
    expected = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), Ry]])
    assert np.allclose(M, expected, atol=1e-12)


# ---------- code-level ----------


def test_dct4_code_returns_callable_and_tensors():
    code, tensors = dct4_code(2, 2)
    assert callable(code)
    assert len(tensors) > 0


def test_dct4_code_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        dct4_code(0, 1)
    with pytest.raises(ValueError):
        dct4_code(1, 0)


def test_dct4_forward_matches_bitreversed_dct_iv():
    # Defining property: per dimension the circuit equals the bit-reversed
    # orthonormal DCT-IV (matching qft_code's bit-reversed output order).
    for m, n in [(1, 2), (2, 2), (2, 3), (3, 3)]:
        code, tensors = dct4_code(m, n)
        Rm, Rn = _dct4_op(m), _dct4_op(n)
        rng = np.random.default_rng(m * 10 + n)
        pic = rng.standard_normal((2**m, 2**n))
        out = np.asarray(dct4_ft_mat(tensors, code, m, n, jnp.asarray(pic, jnp.complex128)))
        assert np.allclose(out.imag, 0.0, atol=1e-12)
        assert np.allclose(out.real, Rm @ pic @ Rn.T, atol=1e-12)


def test_dct4_ft_ift_roundtrip():
    m, n = 2, 3
    code_fwd, tensors = dct4_code(m, n)
    code_inv, _ = dct4_code(m, n, inverse=True)
    rng = np.random.default_rng(0)
    pic = jnp.asarray(rng.standard_normal((2**m, 2**n)), jnp.complex128)
    fwd = dct4_ft_mat(tensors, code_fwd, m, n, pic)
    rec = dct4_ift_mat([jnp.conj(t) for t in tensors], code_inv, m, n, fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


# ---------- basis class ----------


def test_dct4_basis_roundtrip_and_real():
    for m, n in [(2, 3), (3, 3), (4, 4)]:
        b = DCT4Basis(m, n)
        rng = np.random.default_rng(m + n)
        pic = jnp.asarray(rng.standard_normal((2**m, 2**n)), jnp.complex128)
        fwd = b.forward_transform(pic)
        assert np.allclose(np.asarray(fwd).imag, 0.0, atol=1e-12)
        rec = b.inverse_transform(fwd)
        assert jnp.allclose(rec, pic, atol=1e-10)


def test_dct4_gates_are_at_most_two_qubit():
    gates = _dct4_gates_1d(4, offset=0)
    assert all(len(g["qubits"]) <= 2 for g in gates)
    assert {g["kind"] for g in gates} <= {"H", "CP", "U4"}


def _rand_pic_c(m, n, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n))
    return jnp.asarray(a, dtype=jnp.complex128)


def test_controlled_matches_o4_at_init():
    m, n = 3, 3
    pic = _rand_pic_c(m, n)
    b_o4 = DCT4Basis(m, n)
    b_ctl = DCT4Basis(m, n, parametrization="controlled")
    out_o4 = dct4_ft_mat(b_o4.tensors, b_o4.code, m, n, pic)
    out_ctl = dct4_ft_mat(b_ctl.tensors, b_ctl.code, m, n, pic)
    assert jnp.allclose(out_o4, out_ctl, atol=1e-10)


def test_controlled_roundtrips():
    m, n = 3, 2
    pic = _rand_pic_c(m, n)
    b = DCT4Basis(m, n, parametrization="controlled")
    fwd = b.forward_transform(pic)
    back = b.inverse_transform(fwd)
    assert jnp.allclose(back, pic, atol=1e-9)


def test_controlled_drops_twiddle_o4_gates():
    m, n = 3, 3
    b_o4 = DCT4Basis(m, n)
    b_ctl = DCT4Basis(m, n, parametrization="controlled")

    def n4(b):
        return sum(1 for t in b.tensors if t.shape == (2, 2, 2, 2))

    assert n4(b_o4) - n4(b_ctl) == 6


def test_controlled_twiddle_on_o2_manifold():
    from pdft.manifolds import Unitary2qManifold, group_by_manifold
    b_o4 = DCT4Basis(3, 3)
    b_ctl = DCT4Basis(3, 3, parametrization="controlled")

    def n_o4(b):
        g = group_by_manifold(list(b.tensors))
        return sum(len(v) for k, v in g.items() if isinstance(k, Unitary2qManifold))

    assert n_o4(b_o4) - n_o4(b_ctl) == 6


def test_controlled_gradient_flows_to_2x2_leaves():
    import jax
    m, n = 2, 2
    pic = _rand_pic_c(m, n)
    b = DCT4Basis(m, n, parametrization="controlled")
    w = jnp.arange(2 ** (m + n), dtype=jnp.float64).reshape((2**m, 2**n))

    def loss(tensors):
        out = dct4_ft_mat(tensors, b.code, m, n, pic)
        return jnp.sum((jnp.abs(out) ** 2) * w)

    grads = jax.grad(loss)(b.tensors)
    from pdft.bases.circuit.dct4 import _dct4_gates_1d
    from pdft.circuit.builder import sorted_gate_program
    gates = (_dct4_gates_1d(m, offset=0, parametrization="controlled")
             + _dct4_gates_1d(n, offset=m, parametrization="controlled"))
    program = sorted_gate_program(gates)
    cry_idx = [i for i, (kind, _q) in enumerate(program) if kind == "CRY"]
    assert cry_idx, "expected CRY twiddle leaves in the controlled parametrization"
    assert all(float(jnp.max(jnp.abs(grads[i]))) > 1e-8 for i in cry_idx)
