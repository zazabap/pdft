"""Colour images as a stack of channel planes, and the colour space that makes the stack.

A basis acts on one ``(2**m, 2**n)`` plane. The tasks (``pdft.tasks``) and the
step trainer take a stack of planes, ``(C, H, W)``, run the same basis on every
plane and keep one coefficient budget over all of them, so colour enters as a
change of channels: an invertible ``C x C`` matrix applied at every pixel,
which takes an image laid out ``(..., H, W, C)``, as image files are read, to
its planes ``(..., C, H, W)`` and back.

The matrix is what carries the gain. R, G and B of a photograph move
together, and a budget shared across the planes of a decorrelated space sends
the chroma planes few coefficients. Two spaces are provided: ``YCBCR``, the
luma and two chroma differences of JPEG and video (BT.601), and ``klt``, the
principal axes of a dataset's pixel colours, which is orthonormal, so the
error in its coefficients is the error in RGB. JPEG also halves the chroma
resolution before its transform; under a coefficient budget that costs
quality, so nothing here does it. ``ColorSpace(np.eye(3))`` keeps RGB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


@dataclass(frozen=True, eq=False)
class ColorSpace:
    """An invertible change of channels, applied pixel by pixel.

    ``matrix`` is ``(C, C)``: row ``i`` gives channel ``i`` of the space as a
    combination of the image's channels. ``to_planes`` takes an image
    ``(..., H, W, C)`` to its planes ``(..., C, H, W)``, in the precision the
    image has (integers become float64); ``to_image`` is its inverse.
    """

    matrix: np.ndarray
    inverse: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        matrix = np.array(self.matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"the colour matrix must be square, got shape {matrix.shape}")
        if not (np.isfinite(matrix).all() and np.linalg.cond(matrix) < 1e12):
            raise ValueError("the colour matrix must be finite and invertible")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "inverse", np.linalg.inv(matrix))

    @property
    def channels(self) -> int:
        return self.matrix.shape[0]

    def _mix(self, x: Array, matrix: np.ndarray, axis: int, what: str) -> Array:
        """``matrix`` applied along ``axis`` of ``x``, which is moved last: ``(..., H, W, C)``."""
        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.inexact):
            x = x.astype(jnp.float64)
        if x.ndim < 3 or x.shape[axis] != self.channels:
            raise ValueError(
                f"{what} must have {self.channels} channels on axis {axis}, got shape {x.shape}"
            )
        return jnp.moveaxis(x, axis, -1) @ jnp.asarray(matrix, dtype=x.dtype).T

    def to_planes(self, image: Array) -> Array:
        """``(..., H, W, C)`` image -> ``(..., C, H, W)`` planes of this space."""
        return jnp.moveaxis(self._mix(image, self.matrix, -1, "image"), -1, -3)

    def to_image(self, planes: Array) -> Array:
        """``(..., C, H, W)`` planes of this space -> ``(..., H, W, C)`` image."""
        return self._mix(planes, self.inverse, -3, "planes")


# BT.601 luma and chroma differences, as JPEG uses them, without the offset
# JPEG adds to store chroma unsigned: the chroma planes are centred at zero.
YCBCR = ColorSpace(
    np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ]
    )
)


def klt(images) -> ColorSpace:
    """The principal axes of the pixel colours of ``images`` (``(..., H, W, C)``), largest variance first.

    The Karhunen-Loeve transform of the colours: an orthonormal space whose
    planes are uncorrelated over these pixels, so the inverse is the
    transpose and the error in its coefficients is the error in the image.
    On photographs the first axis is a luminance (all its weights positive)
    with about nine tenths of the variance. The sign of each axis is fixed by
    making its largest weight positive. Fit it on training images.
    """
    pixels = np.asarray(images, dtype=np.float64)
    if pixels.ndim < 3:
        raise ValueError(f"images must be (..., H, W, C), got shape {pixels.shape}")
    pixels = pixels.reshape(-1, pixels.shape[-1])
    variances, axes = np.linalg.eigh(np.atleast_2d(np.cov(pixels.T)))  # 2-D for one channel too
    rows = axes[:, ::-1].T  # eigh is ascending
    signs = np.sign(rows[np.arange(len(rows)), np.abs(rows).argmax(axis=1)])
    return ColorSpace(rows * signs[:, None])


__all__ = ["ColorSpace", "YCBCR", "klt"]
