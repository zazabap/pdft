"""The machinery every circuit basis shares: ``CircuitBasis`` in ``pdft.bases.core``."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import CircuitBasis, bases_allclose
from pdft.bases.circuit.qft import qft_gates
from pdft.bases.core import BasisTransforms
from pdft.circuit.builder import CircuitCode, Program, cp_gate, hadamard_gate

from ..helpers import CIRCUIT_CLASSES, complex_image


@dataclass(init=False)
class _Toy(CircuitBasis):
    """A basis defined in three lines: its gates and one extra static field."""

    label: str
    built = 0

    def __init__(self, m, n, tensors=None, code=None, inv_code=None, label="toy"):
        type(self).built += 1
        self.label = label
        gates = [hadamard_gate(q) for q in range(1, m + n + 1)] + [cp_gate(1, m + 1, 0.4)]
        self._init(gates, m, n, tensors, code, inv_code)


@pytest.mark.parametrize("cls", CIRCUIT_CLASSES)
def test_every_circuit_basis_is_a_circuit_basis(cls):
    basis = cls(m=2, n=2)
    assert isinstance(basis, CircuitBasis) and isinstance(basis, BasisTransforms)
    assert isinstance(basis, pdft.AbstractSparseBasis)
    assert isinstance(basis.program, Program) and (basis.program.m, basis.program.n) == (2, 2)
    assert basis.code == CircuitCode(basis.program)
    assert basis.inv_code == CircuitCode(basis.program, inverse=True)
    assert [f.name for f in fields(basis)][:6] == [
        "m",
        "n",
        "tensors",
        "program",
        "code",
        "inv_code",
    ]


def test_a_subclass_is_a_pytree_without_registering_anything():
    built = _Toy.built
    toy = _Toy(m=1, n=2, label="mine")
    assert _Toy.built == built + 1
    leaves, treedef = jax.tree_util.tree_flatten(toy)
    assert all(leaf is t for leaf, t in zip(leaves, toy.tensors)) and len(leaves) == 4
    moved = jax.tree_util.tree_unflatten(treedef, [leaf + 1 for leaf in leaves])
    # the fields come back, extra ones included, and the constructor did not run again
    assert _Toy.built == built + 1
    assert type(moved) is _Toy and (moved.m, moved.n, moved.label) == (1, 2, "mine")
    assert moved.program == toy.program and moved.code == toy.code
    assert all(jnp.array_equal(a, b + 1) for a, b in zip(moved.tensors, toy.tensors))
    assert moved.image_size == (2, 4) and moved.num_parameters == 16
    x = jnp.asarray(np.random.default_rng(0).standard_normal((2, 4)))
    np.testing.assert_allclose(toy.inverse_transform(toy.forward_transform(x)), x, atol=1e-12)


def test_extra_fields_survive_the_pytree_and_show_in_the_repr():
    basis = pdft.TEBDBasis(m=3, n=2, seed=1)
    again = jax.tree_util.tree_map(lambda t: t, basis)
    assert (again.n_row_gates, again.n_col_gates) == (3, 2) and bases_allclose(again, basis)
    assert repr(basis).startswith("TEBDBasis(m=3, n=2, tensors=[")
    assert repr(basis).endswith("n_row_gates=3, n_col_gates=2)")
    assert "program" not in repr(basis) and "code" not in repr(basis)


def test_only_the_qft_topology_freezes_to_a_blocked_basis():
    assert [c.__name__ for c in CIRCUIT_CLASSES if c.freezes_to_blocked] == [
        "QFTBasis",
        "RichBasis",
        "RealRichBasis",
    ]


def test_a_given_code_replaces_the_circuits_own():
    """The constructor still takes ``code`` and ``inv_code``; a use for it is
    asking for the slice arithmetic of the one-qubit gates."""
    plain = pdft.QFTBasis(m=3, n=2)
    fast = pdft.QFTBasis(
        m=3,
        n=2,
        code=replace(plain.code, slices=True),
        inv_code=replace(plain.inv_code, slices=True),
    )
    assert fast.code.slices and fast.inv_code.slices and fast.code != plain.code
    x = jnp.asarray(np.random.default_rng(1).standard_normal((8, 4)))
    np.testing.assert_allclose(fast.forward_transform(x), plain.forward_transform(x), atol=1e-12)
    np.testing.assert_allclose(fast.inverse_transform(x), plain.inverse_transform(x), atol=1e-12)


def test_a_family_with_no_options_only_names_its_emitter():
    class _Plain(CircuitBasis):
        emit = staticmethod(qft_gates)

    basis = _Plain(2, 1)
    reference = pdft.QFTBasis(2, 1)
    assert basis.program == reference.program and type(basis) is _Plain
    assert all(jnp.array_equal(a, b) for a, b in zip(basis.tensors, reference.tensors))
    given = _Plain(2, 1, tensors=[2 * t for t in reference.tensors])
    assert all(jnp.array_equal(a, 2 * b) for a, b in zip(given.tensors, reference.tensors))
    # the base class itself has no circuit to build
    with pytest.raises(AttributeError, match="emit"):
        CircuitBasis(1, 1)


def test_bases_allclose_compares_type_size_and_tensors():
    a = pdft.QFTBasis(m=2, n=2)
    assert bases_allclose(a, pdft.QFTBasis(m=2, n=2))
    assert not bases_allclose(a, pdft.RichBasis(m=2, n=2))
    assert not bases_allclose(a, pdft.QFTBasis(m=2, n=1))
    assert not bases_allclose(a, pdft.QFTBasis(m=2, n=2, tensors=a.tensors[:-1]))
    nudged = pdft.QFTBasis(m=2, n=2, tensors=[*a.tensors[:-1], a.tensors[-1] + 1e-6])
    assert not bases_allclose(a, nudged) and bases_allclose(a, nudged, atol=1e-5)


@pytest.mark.parametrize(
    "make",
    [
        lambda: pdft.EntangledQFTBasis(m=2, n=2, seed=1),
        lambda: pdft.BlockedBasis(pdft.QFTBasis(m=2, n=1), 1, 1),
    ],
)
def test_with_tensors_swaps_the_tensors_and_keeps_everything_else(make):
    from pdft.bases import with_tensors

    basis = make()
    doubled = with_tensors(basis, [2 * t for t in basis.tensors])
    assert type(doubled) is type(basis) and doubled.image_size == basis.image_size
    assert doubled.code == basis.code and doubled.inv_code == basis.inv_code
    assert jax.tree_util.tree_structure(doubled) == jax.tree_util.tree_structure(basis)
    assert all(jnp.array_equal(a, 2 * b) for a, b in zip(doubled.tensors, basis.tensors))
    # the original is untouched
    assert pdft.bases_allclose(basis, make(), atol=0.0)


def test_the_program_follows_a_code_that_is_passed_in():
    """Rebuilding a basis with another instance's tensors and codes is a pattern
    downstream code uses. The codes define the circuit, so the program has to be
    theirs, not the one the constructor would have compiled by default."""
    controlled = pdft.DCT4Basis(m=2, n=2, parametrization="controlled")
    default = pdft.DCT4Basis(m=2, n=2)
    assert controlled.program != default.program

    rebuilt = pdft.DCT4Basis(
        m=2, n=2, tensors=controlled.tensors, code=controlled.code, inv_code=controlled.inv_code
    )
    assert rebuilt.program == controlled.program and rebuilt.code is controlled.code
    assert rebuilt.program.tensor_indices(kind="CRY") == controlled.program.tensor_indices(
        kind="CRY"
    )
    assert rebuilt.program.tensor_indices(kind="CRY") != []
    x = jnp.asarray(np.random.default_rng(2).standard_normal((4, 4)))
    np.testing.assert_array_equal(rebuilt.forward_transform(x), controlled.forward_transform(x))
    np.testing.assert_array_equal(rebuilt.inverse_transform(x), controlled.inverse_transform(x))

    # a code that is not a CircuitCode says nothing about the circuit
    def opaque(*operands):
        return default.code(*operands)

    wrapped = pdft.DCT4Basis(m=2, n=2, code=opaque)
    assert wrapped.program == default.program and wrapped.code is opaque
    np.testing.assert_array_equal(wrapped.forward_transform(x), default.forward_transform(x))


@pytest.mark.parametrize("cls", [pdft.QFTBasis, pdft.RichBasis, pdft.RealRichBasis, pdft.DCT4Basis])
def test_dataclasses_replace_works_where_every_field_is_a_constructor_argument(cls):
    basis = cls(m=2, n=2)
    doubled = replace(basis, tensors=[2 * t for t in basis.tensors])
    assert type(doubled) is cls and doubled.program == basis.program
    assert doubled.code == basis.code and doubled.inv_code == basis.inv_code
    assert all(jnp.array_equal(a, 2 * b) for a, b in zip(doubled.tensors, basis.tensors))


@pytest.mark.parametrize("cls", CIRCUIT_CLASSES)
def test_a_basis_survives_copy_and_pickle(cls):
    import copy
    import pickle

    basis = cls(m=2, n=2)
    x = jnp.asarray(np.random.default_rng(4).standard_normal((4, 4)))
    for clone in (copy.copy(basis), copy.deepcopy(basis), pickle.loads(pickle.dumps(basis))):
        assert type(clone) is cls and clone.program == basis.program and clone.code == basis.code
        assert bases_allclose(clone, basis, atol=0.0)
        np.testing.assert_array_equal(clone.forward_transform(x), basis.forward_transform(x))


def test_a_subclass_that_is_not_a_dataclass_keeps_its_attributes():
    """The pytree carries every attribute of the instance, declared as a field or not."""

    class _Plainer(CircuitBasis):
        flavour = "default"

        def __init__(self, m, n, tensors=None, code=None, inv_code=None, flavour="cp"):
            self.flavour = flavour
            self.note = ("kept", m)
            self._init(qft_gates(m, n), m, n, tensors, code, inv_code)

    basis = _Plainer(2, 2, flavour="u4")
    mapped = jax.tree_util.tree_map(lambda t: 2 * t, basis)
    assert (mapped.flavour, mapped.note) == ("u4", ("kept", 2))
    trained = pdft.train_basis(
        basis,
        target=jnp.asarray(np.random.default_rng(0).standard_normal((4, 4))),
        loss=pdft.L1Norm(),
        optimizer=pdft.RiemannianAdam(lr=0.01),
        steps=2,
    ).basis
    assert type(trained) is _Plainer and (trained.flavour, trained.note) == ("u4", ("kept", 2))
    assert jax.tree_util.tree_structure(mapped) == jax.tree_util.tree_structure(basis)
    assert jax.tree_util.tree_structure(_Plainer(2, 2)) != jax.tree_util.tree_structure(basis)


def test_the_inverse_follows_a_code_passed_alone():
    """A code passed without its inverse still gets the inverse of its own circuit."""
    front = pdft.EntangledQFTBasis(m=2, n=2, seed=1, entangle_position="front")
    rebuilt = pdft.EntangledQFTBasis(m=2, n=2, tensors=front.tensors, code=front.code)
    assert rebuilt.program == front.program and rebuilt.inv_code == front.inv_code
    x = complex_image((4, 4))
    np.testing.assert_allclose(
        rebuilt.inverse_transform(rebuilt.forward_transform(x)), x, atol=1e-12
    )
