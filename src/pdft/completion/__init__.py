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

    transform      U(theta) as a gate circuit; analysis / synthesis pair
    solver         the unrolled IHT map (hard / soft, straight-through)
    unroll         memory-bounded differentiation (nested rematerialisation)
    training       task-adapted training and the shared Adam loop
    adam           plain Adam, written out (this package does not use optax)
    coherence      mu for closures and matrices; certify_flat_modulus
    metrics        PSNR / SSIM / MS-SSIM
    protocol       the evaluation protocol and Table I constants
    data           image loading and the DIV2K / Kodak splits
    bridge         QFTBasis <-> angles / {"g", "phi"} conversion
    families/      general (QFT + diagonals / rotations), shared, butterfly,
                   riemannian (free unitary)
    baselines/     fixed bases, nuclear norm, tensor train, transform learning

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
from .solver import hard_k, iht, reconstruct, reconstruct_batch, soft_k
from .training import init_params, train
from .transform import (
    analysis,
    apply_u,
    coherence,
    gate_pairs,
    n_params,
    synthesis,
    theta0,
    unitary_matrix,
)
from .unroll import estimate_peak, plan, report, split_k
from .unroll import reconstruct as reconstruct_bounded

__all__ = [
    "analysis",
    "angles_from_qft_basis",
    "apply_u",
    "bitrev_image",
    "certify_flat_modulus",
    "coherence",
    "dense_operator",
    "estimate_peak",
    "gate_pairs",
    "general_from_qft_basis",
    "hard_k",
    "iht",
    "init_params",
    "is_flat_modulus",
    "kodak_split",
    "load_gray",
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
    "soft_k",
    "split_k",
    "ssim",
    "synthesis",
    "table1_split",
    "table1_test",
    "table1_val",
    "theta0",
    "train",
    "unitary_matrix",
]
