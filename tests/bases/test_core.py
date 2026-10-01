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

CIRCUIT_BASES = [
    pdft.QFTBasis,
    pdft.EntangledQFTBasis,
    pdft.TEBDBasis,
    pdft.MERABasis,
    pdft.DCT4Basis,
    pdft.RichBasis,
    pdft.RealRichBasis,
]


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


@pytest.mark.parametrize("cls", CIRCUIT_BASES)
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


@pytest.mark.parametrize("cls", CIRCUIT_BASES)
def test_the_program_describes_the_stored_tensors(cls):
    """One step per tensor, and the kind of each step fits the tensor in its slot."""
    basis = cls(m=2, n=2)
    steps = basis.program.sorted_steps
    assert len(steps) == len(basis.tensors)
    for (kind, qubits), tensor in zip(steps, basis.tensors):
        assert tensor.shape == ((2, 2, 2, 2) if kind == "U4" else (2, 2))
        assert len(qubits) == (1 if kind == "H" else 2)
        assert all(1 <= q <= 4 for q in qubits)


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
    assert [c.__name__ for c in CIRCUIT_BASES if c.freezes_to_blocked] == [
        "QFTBasis",
        "RichBasis",
        "RealRichBasis",
    ]
    with pytest.raises(TypeError, match="freeze_as_blocked supports"):
        pdft.freeze_as_blocked(pdft.TEBDBasis(m=2, n=2), 1, 1)


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


@pytest.mark.parametrize("cls", [pdft.TEBDBasis, pdft.MERABasis])
def test_layered_bases_seed_one_phase_per_gate(cls):
    seeded = cls(m=2, n=4, seed=7)
    count = seeded.n_row_gates + seeded.n_col_gates
    drawn = list(np.random.default_rng(7).normal(0.0, 0.1, count))
    assert bases_allclose(seeded, cls(m=2, n=4, phases=drawn), atol=0.0)
    assert not bases_allclose(seeded, cls(m=2, n=4))
    # explicit phases win over the seed
    assert bases_allclose(cls(m=2, n=4, phases=drawn, seed=99), seeded, atol=0.0)
    assert type(seeded).__name__ in repr(seeded) and seeded == seeded
