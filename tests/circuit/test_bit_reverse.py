"""`bit_reverse`: the pixel frame of the image, and the frame the QFT basis works in."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.circuit import bit_reverse, register_width

from ..helpers import complex_image


def test_bit_reverse_permutes_each_index_by_reversing_its_bits():
    x = jnp.arange(8 * 4).reshape(8, 4)
    rows = [0, 4, 2, 6, 1, 5, 3, 7]
    columns = [0, 2, 1, 3]
    np.testing.assert_array_equal(bit_reverse(x), np.asarray(x)[np.ix_(rows, columns)])
    np.testing.assert_array_equal(bit_reverse(bit_reverse(x)), x)
    assert bit_reverse(x).dtype == x.dtype


def test_bit_reverse_carries_batch_axes():
    stack = complex_image((3, 2, 4, 8))
    out = bit_reverse(stack)
    assert out.shape == stack.shape
    for i in range(3):
        for j in range(2):
            np.testing.assert_array_equal(out[i, j], bit_reverse(stack[i, j]))


def test_bit_reverse_needs_power_of_two_sides():
    with pytest.raises(ValueError, match="power-of-two number of values, got 6"):
        bit_reverse(jnp.zeros((4, 6)))
    assert [register_width(size) for size in (1, 2, 8, 1024)] == [0, 1, 3, 10]
    for size in (0, 3, 12):
        with pytest.raises(ValueError, match="power-of-two"):
            register_width(size)


@pytest.mark.parametrize(("m", "n"), [(3, 2), (2, 4), (1, 3)])
def test_the_qft_basis_is_the_dft_in_the_bit_reversed_frame(m, n):
    """The sign convention too: ``e^{+2 pi i k x / N}``, numpy's inverse transform."""
    basis = pdft.QFTBasis(m=m, n=n)
    x = complex_image(basis.image_size, seed=m + n)
    dft = np.fft.ifft2(np.asarray(x), norm="ortho")
    np.testing.assert_allclose(basis.forward_transform(bit_reverse(x)), dft, atol=1e-13)
    np.testing.assert_allclose(
        bit_reverse(basis.inverse_transform(jnp.asarray(dft))), x, atol=1e-13
    )
    # without the adapter it is the transform of the permuted image, not of the image
    assert float(jnp.max(jnp.abs(basis.forward_transform(x) - dft))) > 0.1
