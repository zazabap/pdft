"""What every basis has in common, asserted once over the whole case registry.

The per-basis test files check most of this one class at a time. The refactor
replaces the shared machinery of all of them at once, so these run the same
assertions over every registered case, including the ones no per-basis file
covers.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.bases import bases_allclose
from pdft.bases.block.block import BlockCode

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


def _with_slices(code):
    """The same code with the slice arithmetic for its one-qubit gates."""
    if isinstance(code, BlockCode):
        return replace(code, inner=_with_slices(code.inner))
    return replace(code, slices=True)


@pytest.mark.parametrize("case", CASES)
def test_slice_arithmetic_matches_the_default(case):
    """The opt-in arithmetic changes how one-qubit gates are computed, not what."""
    rng = case_rng(case)
    basis = generic(BASES[case](), rng)
    pic = jnp.asarray(complex_normal(rng, basis.image_size)).reshape((2,) * (basis.m + basis.n))
    conj = [jnp.conj(t) for t in basis.tensors]
    for code, tensors in ((basis.code, basis.tensors), (basis.inv_code, conj)):
        np.testing.assert_allclose(
            _with_slices(code)(*tensors, pic), code(*tensors, pic), rtol=1e-12, atol=1e-12
        )


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


# Two ways a transform reaches its circuit, both as on `main`. The strict
# bases check the image's shape and work in double precision. Rich, RealRich
# and Blocked do neither: any image with the right number of elements is
# reshaped, and the precision is whatever the tensors and the image promote to.
LOOSE = (pdft.RichBasis, pdft.RealRichBasis, pdft.BlockedBasis)


def _single_precision(basis):
    leaves, treedef = jax.tree_util.tree_flatten(basis)
    return jax.tree_util.tree_unflatten(treedef, [leaf.astype(jnp.complex64) for leaf in leaves])


@pytest.mark.parametrize("case", CASES)
def test_transforms_refuse_another_image_size(case):
    basis = BASES[case]()
    rows, cols = basis.image_size
    refusal = TypeError if isinstance(basis, LOOSE) else ValueError
    for transform in (basis.forward_transform, basis.inverse_transform):
        with pytest.raises(refusal):
            transform(jnp.zeros((rows * 2, cols)))


@pytest.mark.parametrize("case", CASES)
def test_images_of_another_shape_with_the_right_size(case):
    basis = BASES[case]()
    rows, cols = basis.image_size
    image = jnp.asarray(case_rng(case).standard_normal((rows, cols)))
    expected = basis.forward_transform(image)
    others = [image.reshape(-1), image[None]]
    if rows != cols:
        others.append(image.reshape(cols, rows))
    for other in others:
        if isinstance(basis, LOOSE):
            np.testing.assert_array_equal(basis.forward_transform(other), expected)
        else:
            with pytest.raises(ValueError, match="pic shape must be"):
                basis.forward_transform(other)


@pytest.mark.parametrize("case", CASES)
def test_precision_of_the_transforms(case):
    """Double-precision tensors give complex128 whatever the image. Single-precision
    tensors stay single precision through the loose bases and are promoted by the
    strict ones, which cast the image."""
    basis = BASES[case]()
    single = _single_precision(basis)
    real = case_rng(case).standard_normal(basis.image_size)
    for image_dtype in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        image = jnp.asarray(real, dtype=image_dtype)
        narrow = image_dtype in (jnp.float32, jnp.complex64)
        for transform in ("forward_transform", "inverse_transform"):
            assert getattr(basis, transform)(image).dtype == jnp.complex128
            wanted = jnp.complex64 if isinstance(basis, LOOSE) and narrow else jnp.complex128
            assert getattr(single, transform)(image).dtype == wanted, (transform, image_dtype)


@pytest.mark.parametrize("case", CASES)
def test_the_loss_keeps_the_precision_of_its_operands(case):
    """`loss_function` never casts: single-precision tensors and image give a
    single-precision loss and gradient, for every basis."""
    basis = BASES[case]()
    single = _single_precision(basis)
    m, n = basis.m, basis.n
    image = jnp.asarray(case_rng(case).standard_normal(basis.image_size), dtype=jnp.float32)
    for loss in (pdft.L1Norm(), pdft.MSELoss(k=3)):

        def value(tensors, b, loss=loss):
            return pdft.loss_function(tensors, m, n, b.code, image, loss, inverse_code=b.inv_code)

        assert value(list(single.tensors), single).dtype == jnp.float32
        assert value(list(basis.tensors), basis).dtype == jnp.float64
        gradient = jax.grad(value)(list(single.tensors), single)
        assert all(g.dtype == jnp.complex64 for g in gradient)
    with pytest.raises(ValueError, match="pic shape must be"):
        pdft.loss_function(list(basis.tensors), m, n, basis.code, image.reshape(-1), pdft.L1Norm())
