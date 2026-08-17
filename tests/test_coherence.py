import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.bases import (
    EntangledQFTBasis,
    QFTBasis,
    RealRichBasis,
    RichBasis,
    TEBDBasis,
)
from pdft.circuit.builder import controlled_phase_diag, is_compact_cp
from pdft.coherence import (
    certify_flat_modulus,
    coherence,
    dense_operator,
    diagonal_tensor_indices,
    is_flat_modulus,
)

# (3, 3) keeps the dense 64x64 operator cheap while exercising both registers.
M = N = 3
ALL_BASES = [QFTBasis, EntangledQFTBasis, TEBDBasis, RichBasis, RealRichBasis]


def _rand_unitary(key, d):
    a = jax.random.normal(key, (d, d)) + 1j * jax.random.normal(key, (d, d))
    q, r = jnp.linalg.qr(a)
    return q * (jnp.diag(r) / jnp.abs(jnp.diag(r)))


def _perturb(basis, key, *, diagonal: bool):
    """Randomise either the diagonal (CP) tensors or the rest, in place."""
    leaves, treedef = jax.tree_util.tree_flatten(basis)
    out = []
    for i, t in enumerate(leaves):
        k = jax.random.fold_in(key, i)
        cp = t.shape == (2, 2) and is_compact_cp(t)
        if diagonal and cp:
            phi = float(jax.random.uniform(k, (), minval=-jnp.pi, maxval=jnp.pi))
            out.append(controlled_phase_diag(phi).astype(t.dtype))
        elif not diagonal and not cp:
            d = round(float(np.prod(t.shape)) ** 0.5)
            out.append(_rand_unitary(k, d).reshape(t.shape).astype(t.dtype))
        else:
            out.append(t)
    return jax.tree_util.tree_unflatten(treedef, out)


# --------------------------------------------------------------------------
# dense_operator


def test_dense_operator_is_unitary():
    u = dense_operator(QFTBasis(m=M, n=N))
    dim = 2**M * 2**N
    assert u.shape == (dim, dim)
    assert jnp.allclose(jnp.conj(u).T @ u, jnp.eye(dim), atol=1e-10)


def test_dense_operator_columns_match_forward_transform():
    b = QFTBasis(m=2, n=2)
    u = dense_operator(b)
    pic = jnp.zeros((4, 4), dtype=jnp.complex128).at[1, 2].set(1.0)
    assert jnp.allclose(u[:, 1 * 4 + 2], b.forward_transform(pic).reshape(-1), atol=1e-12)


# --------------------------------------------------------------------------
# the guarantee


@pytest.mark.parametrize("ctor", ALL_BASES)
def test_mu_is_one_at_initialisation(ctor):
    assert coherence(ctor(m=M, n=N)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("ctor", ALL_BASES)
def test_mu_is_one_for_every_diagonal_parameter_value(ctor):
    """Proposition: freeing only the diagonal gates cannot move mu."""
    base = ctor(m=M, n=N)
    for seed in range(5):
        perturbed = _perturb(base, jax.random.PRNGKey(seed), diagonal=True)
        assert coherence(perturbed) == pytest.approx(1.0, abs=1e-9)
        assert is_flat_modulus(perturbed)


@pytest.mark.parametrize("ctor", ALL_BASES)
def test_freeing_the_non_diagonal_gates_breaks_it(ctor):
    """The converse: the guarantee is about which gates are free, not luck."""
    base = ctor(m=M, n=N)
    worst = max(
        coherence(_perturb(base, jax.random.PRNGKey(100 + s), diagonal=False)) for s in range(5)
    )
    assert worst > 1.5, f"expected mu to rise well above 1, got {worst}"


@pytest.mark.parametrize("ctor", ALL_BASES)
def test_mu_stays_within_its_bounds(ctor):
    dim = 2**M * 2**N
    for diagonal in (True, False):
        mu = coherence(_perturb(ctor(m=M, n=N), jax.random.PRNGKey(7), diagonal=diagonal))
        assert 1.0 - 1e-9 <= mu <= dim + 1e-9


# --------------------------------------------------------------------------
# certificate


def test_certificate_holds_when_only_diagonal_gates_train():
    b = QFTBasis(m=M, n=N)
    diagonal = set(diagonal_tensor_indices(b))
    frozen = [i for i in range(len(b.tensors)) if i not in diagonal]
    cert = certify_flat_modulus(b, frozen_indices=frozen)
    assert cert and cert.holds
    assert cert.offending_indices == []
    assert cert.mu == pytest.approx(1.0, abs=1e-9)


def test_certificate_fails_and_names_the_gates_to_freeze():
    b = QFTBasis(m=M, n=N)
    cert = certify_flat_modulus(b)  # nothing frozen: the Hadamards are trainable
    assert not cert
    assert cert.offending_indices == sorted(
        set(range(len(b.tensors))) - set(diagonal_tensor_indices(b))
    )
    assert "freeze indices" in cert.reason


def test_certificate_offending_indices_are_exactly_what_must_be_frozen():
    b = QFTBasis(m=M, n=N)
    cert = certify_flat_modulus(b)
    assert certify_flat_modulus(b, frozen_indices=cert.offending_indices).holds


def test_rich_basis_has_no_diagonal_gates_so_cannot_be_certified():
    """RichBasis frees every gate, so no subset of it is phase-only."""
    b = RichBasis(m=M, n=N)
    assert diagonal_tensor_indices(b) == []
    assert not certify_flat_modulus(b)


def test_certificate_is_falsy_when_the_basis_is_not_flat():
    b = _perturb(QFTBasis(m=M, n=N), jax.random.PRNGKey(3), diagonal=False)
    cert = certify_flat_modulus(b, frozen_indices=list(range(len(b.tensors))))
    assert not cert
    assert "not flat-modulus" in cert.reason
