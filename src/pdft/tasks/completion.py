"""Image completion: fill in an image from the pixels that were observed.

Not in upstream, so there is no Julia behaviour to match; the reference is the
code of the completion paper, and ``tests/parity/test_completion.py`` holds
this module to a run of it.

The image is modelled as ``k``-sparse in the basis and recovered by iterative
hard thresholding: from the zero-filled observation, alternate sparsity in the
basis with consistency on the observed pixels,

    X <- P Y + (1 - P) Re T^-1( H_k( T X ) ),

with ``T`` the basis, ``H_k`` the ``k`` largest coefficients and ``P`` the
observation mask. A fixed number of steps of that is a differentiable map from
the tensors to the reconstruction, so a basis can be trained through the
solver at the sampling rate it will be used at: ``completion_loss`` is that
objective and ``pdft.training.train_basis_steps`` the trainer.

Everything here is in the basis's own frame, as ``compress`` is. A mask lives
on the pixels, so for a basis whose frame is not the image's (the QFT
topology: see ``pdft.circuit.bit_reverse``) reverse the observation and the
mask going in and the reconstruction coming out.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ..loss import topk_truncate

Array = jax.Array


def complete(basis, observed: Array, mask: Array, *, k: int, steps: int) -> Array:
    """Reconstruct a real image from its pixels under ``mask``.

    Parameters
    ----------
    basis
        Any basis: only ``forward_transform`` and ``inverse_transform`` are called.
    observed : Array
        The image, of the basis's ``image_size``. Only the pixels under
        ``mask`` are read.
    mask : Array
        Boolean, same shape: ``True`` where the pixel was observed.
    k : int
        Coefficients kept at each step.
    steps : int
        Solver iterations.

    Returns
    -------
    Array
        The reconstruction: equal to ``observed`` under the mask, real, in the
        precision the basis's transforms return.

    Traceable in the basis's tensors, ``observed`` and ``mask``; ``k`` and
    ``steps`` are static. Each step is rematerialised in the backward pass, so
    a gradient holds one image per step, not the intermediates of every gate.

    The coefficients of a real image under a Fourier-like basis come in pairs
    of equal magnitude, and a cut between the two of a pair would be decided
    by rounding. Magnitudes within ``sqrt(eps)`` of the cut count as tied and
    the tie is settled by position (``topk_truncate``'s ``rtol``), so the
    result does not depend on the device or on the arithmetic of the applier.
    """
    if k < 1 or steps < 1:
        raise ValueError(f"k and steps must be positive, got k={k}, steps={steps}")
    if observed.shape != basis.image_size or mask.shape != basis.image_size:
        raise ValueError(
            f"observed and mask must have the basis's image size {basis.image_size}, "
            f"got {observed.shape} and {mask.shape}"
        )
    if jnp.iscomplexobj(observed):
        raise ValueError("completion reconstructs a real image; observed is complex")
    mask = jnp.asarray(mask, dtype=bool)
    zero_filled = jnp.where(mask, observed, 0.0)

    def step(x: Array, _) -> tuple[Array, None]:
        coefficients = basis.forward_transform(x)
        tie_band = float(jnp.finfo(coefficients.dtype).eps) ** 0.5
        sparse = topk_truncate(coefficients, k, rtol=tie_band)
        return jnp.where(mask, zero_filled, jnp.real(basis.inverse_transform(sparse))), None

    # The iterate has the precision the transforms return, which need not be the image's.
    precision = jax.eval_shape(lambda x: step(x, None)[0], zero_filled).dtype
    return jax.lax.scan(jax.checkpoint(step), zero_filled.astype(precision), None, length=steps)[0]


def completion_loss(*, k: int, steps: int) -> Callable[[object, Array, Array], Array]:
    """``(basis, images, masks) -> scalar``: the mean squared error of ``complete`` over a batch.

    The objective of training through the solver. ``images`` and ``masks`` are
    stacks along a leading axis; the unobserved pixels are the supervision and
    are never shown to the solver.
    """

    def objective(basis, images: Array, masks: Array) -> Array:
        def solve(image: Array, mask: Array) -> Array:
            return complete(basis, image, mask, k=k, steps=steps)

        return jnp.mean((jax.vmap(solve)(images, masks) - images) ** 2)

    return objective
