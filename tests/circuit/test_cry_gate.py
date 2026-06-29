import jax.numpy as jnp
import numpy as np

from pdft.bases.circuit.dct4 import _cry_u4, _ry
from pdft.circuit.builder import Gate, apply_circuit, compile_circuit


def _rand_pic(m, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n))
    return jnp.asarray(a, dtype=jnp.complex128)


def test_cry_matches_cry_u4_forward():
    """A CRY gate (2x2 leaf, applied controlled) equals the dense U4 CR_y."""
    theta = 0.37
    m, n = 1, 1  # qubit 1 = control, qubit 2 = target
    pic = _rand_pic(m, n)
    code_cry, t_cry = compile_circuit(
        [Gate(kind="CRY", qubits=(1, 2), tensor=_ry(theta), phase=0.0)], m, n, inverse=False)
    code_u4, t_u4 = compile_circuit(
        [Gate(kind="U4", qubits=(1, 2), tensor=_cry_u4(theta), phase=0.0)], m, n, inverse=False)
    out_cry = apply_circuit(t_cry, code_cry, m, n, pic)
    out_u4 = apply_circuit(t_u4, code_u4, m, n, pic)
    assert jnp.allclose(out_cry, out_u4, atol=1e-10)


def test_cry_roundtrip_identity():
    """forward then inverse (with real R_y) reconstructs the input."""
    theta = 0.81
    m, n = 1, 2
    pic = _rand_pic(m, n)
    gates = [Gate(kind="CRY", qubits=(1, 3), tensor=_ry(theta), phase=0.0)]
    code_f, tf = compile_circuit(gates, m, n, inverse=False)
    code_i, ti = compile_circuit(gates, m, n, inverse=True)
    fwd = apply_circuit(tf, code_f, m, n, pic)
    back = apply_circuit([jnp.conj(t) for t in ti], code_i, m, n, fwd)
    assert jnp.allclose(back, pic, atol=1e-10)
