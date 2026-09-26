"""pdft.completion: image inpainting from random pixels with a trainable QFT.

Ported from the code of *Image Inpainting from Random Pixels with a Trainable
Quantum Fourier Transform* (github.com/zazabap/pdft-completion). The claim:
recovery from pointwise samples is governed by incoherence with the pixel
basis rather than by sparsity, so the transform is trained *through* the
recovery solver while its coherence stays pinned at the theoretical minimum
for every parameter value (Proposition 1; see :mod:`pdft.coherence`).

The subpackage carries the QFT circuit in a second representation --- the
gate angles, applied directly to the image in O(N log N) with no matrix ever
formed, trained by plain Adam --- next to the core package's tensor lists on
Riemannian manifolds. :mod:`pdft.completion.bridge` converts between the two
exactly. Everything here is a library: the paper's experiment scripts, data
fetchers, figures and result files stay in the paper's repository.

Layout::

    transform      the gate kernel, the per-axis operators, and ``separable``,
                   the 2-D analysis / synthesis pair of any of them
    solver         the unrolled IHT map, and ``solver_for``: the jitted K-step
                   solver of any per-axis operator
    unroll         memory-bounded differentiation (nested rematerialisation)
    training       the minibatch schedule, the task loss of any solver, the
                   Adam loop; every trained family goes through these
    adam           plain Adam, written out (this package does not use optax)
    coherence      mu for closures and matrices; certify_flat_modulus
    metrics        PSNR / SSIM / MS-SSIM
    protocol       the evaluation protocol and Table I constants
    data           image loading and the DIV2K / Kodak splits
    bridge         QFTBasis <-> angles / {"g", "phi"} conversion
    families/      general (QFT + diagonals / rotations), shared, butterfly,
                   riemannian (free unitary)
    baselines/     fixed bases, nuclear norm, tensor train, transform learning

A family is one per-axis operator ``apply(x, params, adjoint, axis)``; the
2-D pair, the solver, the batched solver and the evaluation come from
``separable``, ``solver_for``, ``batched`` and ``evaluate_params``. Register
widths are read off array shapes, never passed.

Importing this subpackage assumes ``pdft`` has enabled JAX x64 mode, which
``import pdft`` does.
"""

from .bridge import (
    angles_from_qft_basis,
    bitrev_image,
    general_from_qft_basis,
    qft_basis_from_angles,
    qft_basis_from_general,
)
from .coherence import certify_flat_modulus, dense_operator, is_flat_modulus
from .data import kodak_split, load_gray, table1_split, table1_test, table1_val
from .metrics import ms_ssim, psnr, score, ssim
from .protocol import evaluate, evaluate_params
from .solver import batched, hard_k, iht, reconstruct, reconstruct_batch, soft_k, solver_for
from .training import adam_loop, init_params, minibatches, task_loss, train
from .transform import (
    analysis,
    apply_dense,
    apply_gates,
    apply_u,
    coherence,
    gate_pairs,
    n_params,
    separable,
    synthesis,
    theta0,
    theta_to_params,
    unitary_matrix,
)
from .unroll import estimate_peak, plan, report, split_k
from .unroll import reconstruct as reconstruct_bounded

__all__ = [
    "adam_loop",
    "analysis",
    "angles_from_qft_basis",
    "apply_dense",
    "apply_gates",
    "apply_u",
    "batched",
    "bitrev_image",
    "certify_flat_modulus",
    "coherence",
    "dense_operator",
    "estimate_peak",
    "evaluate",
    "evaluate_params",
    "gate_pairs",
    "general_from_qft_basis",
    "hard_k",
    "iht",
    "init_params",
    "is_flat_modulus",
    "kodak_split",
    "load_gray",
    "minibatches",
    "ms_ssim",
    "n_params",
    "plan",
    "psnr",
    "qft_basis_from_angles",
    "qft_basis_from_general",
    "reconstruct",
    "reconstruct_batch",
    "reconstruct_bounded",
    "report",
    "score",
    "separable",
    "soft_k",
    "solver_for",
    "split_k",
    "ssim",
    "synthesis",
    "table1_split",
    "table1_test",
    "table1_val",
    "task_loss",
    "theta0",
    "theta_to_params",
    "train",
    "unitary_matrix",
]
