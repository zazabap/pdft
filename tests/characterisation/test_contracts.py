"""What every basis has in common, asserted once over the whole case registry.

The per-basis test files check most of this one class at a time. The refactor
replaces the shared machinery of all of them at once, so these run the same
assertions over every registered case, including the ones no per-basis file
covers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.bases import bases_allclose

from .cases import BASES, case_rng, complex_normal, generic

CASES = list(BASES)


@pytest.mark.parametrize("case", CASES)
def test_public_surface(case):
    basis = BASES[case]()
    rows, cols = basis.image_size
    assert (rows, cols) == (2**basis.m, 2**basis.n)
    assert basis.num_parameters == sum(int(t.size) for t in basis.tensors)
    assert basis.inv_tensors is basis.tensors or all(
        a is b for a, b in zip(basis.inv_tensors, basis.tensors)
    )
    assert callable(basis.code) and callable(basis.inv_code)
    assert repr(basis).startswith(type(basis).__name__ + "(")


@pytest.mark.parametrize("case", CASES)
def test_pytree_leaves_are_the_tensors_in_order(case):
    """The contract ``train_basis`` relies on: leaves are the tensor list, and
    unflattening other leaves gives the same basis carrying them."""
    basis = BASES[case]()
    leaves, treedef = jax.tree_util.tree_flatten(basis)
    assert len(leaves) == len(basis.tensors)
    assert all(leaf is t for leaf, t in zip(leaves, basis.tensors))
    doubled = jax.tree_util.tree_unflatten(treedef, [2 * leaf for leaf in leaves])
    assert type(doubled) is type(basis) and (doubled.m, doubled.n) == (basis.m, basis.n)
    assert all(jnp.array_equal(d, 2 * t) for d, t in zip(doubled.tensors, basis.tensors))
    assert bases_allclose(jax.tree_util.tree_map(lambda t: t, basis), basis)
    # The rebuilt basis computes with the leaves it was given, not the ones it was built from.
    probe = jnp.asarray(complex_normal(case_rng(case), basis.image_size))
    shape = (2,) * (basis.m + basis.n)
    np.testing.assert_allclose(
        doubled.forward_transform(probe),
        basis.code(*doubled.tensors, probe.reshape(shape)).reshape(basis.image_size),
        rtol=1e-12,
    )
    assert not jnp.allclose(doubled.forward_transform(probe), basis.forward_transform(probe))


@pytest.mark.parametrize("case", CASES)
def test_two_instances_are_the_same_basis(case):
    """Same tensors, and the same pytree structure: the code a basis carries
    compares by its program, so a second instance does not retrace or recompile."""
    a, b = BASES[case](), BASES[case]()
    assert bases_allclose(a, b, atol=0.0)
    assert a.code == b.code and a.inv_code == b.inv_code
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


@pytest.mark.parametrize("case", CASES)
def test_code_maps_over_a_stack_of_images(case):
    """The batched trainer vmaps ``code`` over the image; it must equal one call per image."""
    rng = case_rng(case)
    basis = generic(BASES[case](), rng)
    shape = (2,) * (basis.m + basis.n)
    stack = jnp.asarray(complex_normal(rng, (3, *basis.image_size))).reshape((3, *shape))
    for code, tensors in (
        (basis.code, basis.tensors),
        (basis.inv_code, [jnp.conj(t) for t in basis.tensors]),
    ):
        mapped = jax.vmap(lambda x, code=code, tensors=tensors: code(*tensors, x))(stack)
        for i in range(3):
            np.testing.assert_allclose(mapped[i], code(*tensors, stack[i]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_transforms_refuse_another_image_size(case):
    basis = BASES[case]()
    rows, cols = basis.image_size
    with pytest.raises((ValueError, TypeError)):
        basis.forward_transform(jnp.zeros((rows * 2, cols)))
