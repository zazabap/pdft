import jax.numpy as jnp

from pdft.qft import (
    HADAMARD,
    _qft_gates_1d,
    controlled_phase_diag,
    ft_mat,
    ift_mat,
    qft_code,
)


def test_hadamard_is_unitary():
    H = HADAMARD
    assert jnp.allclose(H @ jnp.conj(H).T, jnp.eye(2), atol=1e-12)


def test_controlled_phase_diag_structure():
    phi = 1.2345
    CP = controlled_phase_diag(phi)
    # Compact 2x2 diagonal: only (1, 1) picks up e^{i*phi}
    expected = jnp.array([[1.0 + 0j, 1.0], [1.0, jnp.exp(1j * phi)]], dtype=jnp.complex128)
    assert jnp.allclose(CP, expected, atol=1e-12)


def test_qft_gates_1d_counts():
    gates = _qft_gates_1d(n_qubits=3, offset=0)
    hadamards = [g for g in gates if g["kind"] == "H"]
    cphases = [g for g in gates if g["kind"] == "CP"]
    assert len(hadamards) == 3
    assert len(cphases) == 3  # n(n-1)/2 = 3 for n=3


def test_qft_code_returns_callable_and_tensors():
    code, tensors = qft_code(2, 2)
    # 2 blocks of 2 qubits each: 2 H + 1 CP = 3 gates per block; 6 total.
    assert len(tensors) == 6
    assert callable(code)


def test_qft_code_tensor_count_mxn_matches_formula():
    for (m, n) in [(1, 1), (2, 2), (3, 2), (2, 3), (3, 3)]:
        _, tensors = qft_code(m, n)
        expected = (m + m * (m - 1) // 2) + (n + n * (n - 1) // 2)
        assert len(tensors) == expected


def test_qft_code_rejects_bad_dimensions():
    import pytest

    with pytest.raises(ValueError):
        qft_code(0, 1)
    with pytest.raises(ValueError):
        qft_code(1, 0)


def test_ft_ift_roundtrip_returns_original_to_tolerance():
    m, n = 2, 2
    code_fwd, tensors = qft_code(m, n)
    code_inv, _ = qft_code(m, n, inverse=True)
    pic = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=jnp.complex128,
    )
    fwd = ft_mat(tensors, code_fwd, m, n, pic)
    recovered = ift_mat([jnp.conj(t) for t in tensors], code_inv, m, n, fwd)
    assert jnp.allclose(recovered, pic, atol=1e-10)


def test_ft_mat_of_constant_dc_has_all_energy():
    # Constant input → DC bin dominates after unitary QFT
    m, n = 2, 2
    code_fwd, tensors = qft_code(m, n)
    pic = jnp.ones((4, 4), dtype=jnp.complex128)
    fwd = ft_mat(tensors, code_fwd, m, n, pic)
    assert jnp.isclose(float(jnp.abs(fwd[0, 0])), 16.0 / jnp.sqrt(16.0), atol=1e-8)


def test_ft_mat_rejects_bad_shape():
    import pytest

    code_fwd, tensors = qft_code(2, 2)
    bad_pic = jnp.ones((3, 4), dtype=jnp.complex128)
    with pytest.raises(ValueError):
        ft_mat(tensors, code_fwd, 2, 2, bad_pic)
