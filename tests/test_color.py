"""`pdft.color`: a colour space is a change of channels applied at every pixel, and the two provided."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.color import YCBCR, ColorSpace, klt


def _rgb(shape=(8, 4, 3), seed=0) -> np.ndarray:
    return np.random.default_rng(seed).random(shape)


def _correlated_rgb(count=200, seed=1) -> np.ndarray:
    """Pixels whose channels move together, as a photograph's do, with a little of their own."""
    rng = np.random.default_rng(seed)
    light = rng.random((count, 8, 4, 1))
    return 0.8 * light * np.array([1.0, 0.9, 0.7]) + 0.2 * rng.random((count, 8, 4, 3))


def test_ycbcr_is_the_bt601_luma_and_two_zero_mean_chroma_differences():
    assert YCBCR.channels == 3
    # the luma weights are PIL's "L" conversion; a grey pixel has no chroma
    np.testing.assert_allclose(YCBCR.matrix[0], [0.299, 0.587, 0.114])
    np.testing.assert_allclose(YCBCR.matrix[1:].sum(axis=1), 0.0, atol=1e-6)
    grey = np.full((2, 2, 3), 0.3)
    planes = YCBCR.to_planes(grey)
    np.testing.assert_allclose(planes[0], 0.3)
    np.testing.assert_allclose(planes[1:], 0.0, atol=1e-7)


def test_to_planes_applies_each_row_at_every_pixel_and_to_image_undoes_it():
    rgb = _rgb((2, 8, 4, 3))
    planes = YCBCR.to_planes(rgb)
    assert planes.shape == (2, 3, 8, 4)
    for channel in range(3):
        np.testing.assert_allclose(planes[:, channel], rgb @ YCBCR.matrix[channel], atol=1e-15)
    back = YCBCR.to_image(planes)
    assert back.shape == rgb.shape
    np.testing.assert_allclose(back, rgb, atol=1e-15)
    # one image, no batch axis
    np.testing.assert_allclose(YCBCR.to_image(YCBCR.to_planes(rgb[0])), rgb[0], atol=1e-15)


def test_the_precision_of_the_planes_follows_the_image():
    rgb = _rgb()
    assert YCBCR.to_planes(rgb).dtype == jnp.float64
    assert YCBCR.to_planes(rgb.astype(np.float32)).dtype == jnp.float32
    # integers, as image files are read, become float64 and keep their scale
    bytes_ = (rgb * 255).astype(np.uint8)
    planes = YCBCR.to_planes(bytes_)
    assert planes.dtype == jnp.float64
    np.testing.assert_allclose(planes[0], bytes_ @ YCBCR.matrix[0])


def test_a_colour_space_is_traceable():
    rgb = _rgb()
    np.testing.assert_array_equal(jax.jit(YCBCR.to_planes)(rgb), YCBCR.to_planes(rgb))
    planes = YCBCR.to_planes(rgb)
    np.testing.assert_array_equal(jax.jit(YCBCR.to_image)(planes), YCBCR.to_image(planes))


def test_the_identity_space_only_moves_the_channel_axis():
    rgb = _rgb()
    planes = ColorSpace(np.eye(3)).to_planes(rgb)
    np.testing.assert_array_equal(planes, np.moveaxis(rgb, -1, -3))


def test_klt_is_orthonormal_decorrelates_the_planes_and_orders_them_by_variance():
    rgb = _correlated_rgb()
    space = klt(rgb)
    np.testing.assert_allclose(space.matrix @ space.matrix.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(space.inverse, space.matrix.T, atol=1e-12)
    planes = space.to_planes(rgb)
    covariance = np.cov(np.asarray(planes).transpose(1, 0, 2, 3).reshape(3, -1))
    variances = np.diag(covariance)
    np.testing.assert_allclose(covariance - np.diag(variances), 0.0, atol=1e-12)
    assert variances[0] > variances[1] > variances[2]
    # the first axis is a luminance: every channel enters it with the same sign
    assert np.all(space.matrix[0] > 0)
    # the sign of each axis is fixed by its largest weight
    assert np.all(space.matrix[np.arange(3), np.abs(space.matrix).argmax(axis=1)] > 0)
    np.testing.assert_allclose(space.to_image(planes), rgb, atol=1e-14)


def test_klt_can_be_fitted_on_one_image_and_on_four_channels():
    rgb = _correlated_rgb()
    assert klt(rgb[0]).channels == 3
    four = np.concatenate([rgb, rgb[..., :1] ** 2], axis=-1)
    space = klt(four)
    assert space.channels == 4 and space.to_planes(four).shape == (len(four), 4, 8, 4)
    # one channel: the identity
    np.testing.assert_array_equal(klt(rgb[..., :1]).matrix, [[1.0]])


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 3)), "must be square"),
        (np.ones(3), "must be square"),
        ([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "must be finite and invertible"),
        (np.full((3, 3), np.nan), "must be finite and invertible"),
        (np.diag([1.0, 1.0, np.inf]), "must be finite and invertible"),
    ],
)
def test_a_matrix_that_is_not_a_change_of_channels_is_refused(matrix, message):
    with pytest.raises(ValueError, match=message):
        ColorSpace(matrix)


def test_arrays_with_the_wrong_channels_are_refused():
    rgb = _rgb()
    with pytest.raises(
        ValueError, match=r"image must have 3 channels on axis -1, got shape \(8, 4, 2\)"
    ):
        YCBCR.to_planes(rgb[..., :2])
    with pytest.raises(
        ValueError, match=r"image must have 3 channels on axis -1, got shape \(8, 4\)"
    ):
        YCBCR.to_planes(rgb[..., 0])
    with pytest.raises(
        ValueError, match=r"planes must have 3 channels on axis -3, got shape \(4, 8, 4\)"
    ):
        YCBCR.to_image(np.zeros((4, 8, 4)))
    with pytest.raises(
        ValueError, match=r"planes must have 3 channels on axis -3, got shape \(8, 4\)"
    ):
        YCBCR.to_image(np.zeros((8, 4)))
    with pytest.raises(ValueError, match=r"images must be \(\.\.\., H, W, C\), got shape \(4, 3\)"):
        klt(rgb[0])
