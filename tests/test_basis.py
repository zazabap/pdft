import jax
import jax.numpy as jnp

from pdft.basis import QFTBasis, bases_allclose


def test_qftbasis_constructor_initializes_tensors():
    b = QFTBasis(m=2, n=2)
    # 2 blocks of 2 qubits: 2 H + 1 CP per block, 6 gates total
    assert len(b.tensors) == 6
    # inv_tensors is a back-compat property aliasing tensors
    assert b.inv_tensors is b.tensors
    assert b.m == 2 and b.n == 2


def test_qftbasis_is_jax_pytree():
    b = QFTBasis(m=2, n=2)
    leaves, treedef = jax.tree_util.tree_flatten(b)
    # Julia's QFTBasis stores ONE tensor list; pytree leaves match.
    assert len(leaves) == 6
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert bases_allclose(restored, b)


def test_forward_transform_roundtrips_with_inverse():
    b = QFTBasis(m=2, n=2)
    pic = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=jnp.complex128,
    )
    fwd = b.forward_transform(pic)
    back = b.inverse_transform(fwd)
    assert jnp.allclose(back, pic, atol=1e-10)


def test_num_parameters_matches_total_tensor_elements():
    b = QFTBasis(m=2, n=2)
    expected = sum(t.size for t in b.tensors)
    assert b.num_parameters == expected


def test_image_size():
    assert QFTBasis(m=2, n=3).image_size == (4, 8)


def test_bases_allclose_respects_tolerance():
    b1 = QFTBasis(m=2, n=2)
    b2 = QFTBasis(m=2, n=2, tensors=[t + 1e-14 for t in b1.tensors])
    assert bases_allclose(b1, b2, atol=1e-10)
    assert not bases_allclose(b1, b2, atol=1e-20)


def test_bases_allclose_different_shapes():
    assert not bases_allclose(QFTBasis(m=2, n=2), QFTBasis(m=2, n=3))


def test_qftbasis_rejects_bad_dims():
    import pytest

    with pytest.raises(ValueError):
        QFTBasis(m=0, n=1)
