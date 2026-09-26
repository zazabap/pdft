"""pdft.completion: image inpainting from random pixels with a trainable QFT.

Ported from the code of *Image Inpainting from Random Pixels with a Trainable
Quantum Fourier Transform* (github.com/zazabap/pdft-completion). Recovery from
pointwise samples is governed by incoherence with the pixel basis rather than
by sparsity, so the transform is trained through the recovery solver while its
coherence stays pinned at the theoretical minimum for every parameter value
(Proposition 1; see :mod:`pdft.coherence`).

A transform family is one per-axis operator ``apply(x, params, adjoint, axis)``.
Everything else is derived from it: ``separable`` gives the 2-D analysis and
synthesis pair, ``solver_for`` the jitted K-step solver, ``batched`` the
vmapped one, ``evaluate_params`` the scoring, ``task_loss`` the objective and
``minibatches`` the one batch and mask schedule every trainer draws from.
Register widths are read off array shapes, never passed.

The names below are the ones most used interactively; everything else is
imported from its module. Importing this subpackage assumes ``pdft`` has
enabled JAX x64 mode, which ``import pdft`` does.
"""

from .coherence import certify_flat_modulus
from .transform import apply_gates, separable, theta0

__all__ = [
    "apply_gates",
    "certify_flat_modulus",
    "separable",
    "theta0",
]
