<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/zazabap/pdft/main/docs/_static/logo-dark.svg">
    <img alt="pdft" src="https://raw.githubusercontent.com/zazabap/pdft/main/docs/_static/logo-light.svg" width="340">
  </picture>
</p>

# pdft

[![arXiv](https://img.shields.io/badge/arXiv-2608.00053-b31b1b.svg)](https://arxiv.org/abs/2608.00053)
[![Docs](https://img.shields.io/badge/docs-zazabap.github.io%2Fpdft-blue.svg)](https://zazabap.github.io/pdft/)

A Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl):
learning parametric quantum Fourier transforms via manifold optimization. The
package implements a variational approach that approximates the Discrete
Fourier Transform (DFT) with parameterized quantum circuits.

This is the reference implementation accompanying the paper
[*Fast Trainable Multilinear Bases for Image Compression*](https://arxiv.org/abs/2608.00053)
(An, Ni, Zhou, Liu, 2026). The API reference and example gallery are at
[zazabap.github.io/pdft](https://zazabap.github.io/pdft/).

> Status: feature-complete port. All bases (QFT, entangled QFT, TEBD, MERA,
> Rich/RealRich, DCT-IV, blocked), both Riemannian optimizers (GD + Adam),
> training, JSON/compression I/O, and visualization are implemented, with
> parity against the Julia reference verified by committed goldens.

## Installation

From PyPI (Python 3.11+):

```bash
pip install "pdft>=0.2.3"
```

> **Note:** the older `pdft==0.2.2` wheel predates `DCT4Basis` and the
> `parametrization="u4"` option of `TEBDBasis` / `MERABasis`, so it cannot run
> the paper's DCT-IV, TEBD-U4, or MERA-U4 configurations. If
> `pdft.__version__` reports `0.2.2`, upgrade with `pip install -U pdft`.

From source:

```bash
git clone https://github.com/zazabap/pdft.git
cd pdft
pip install -e ".[dev]"
```

## Quick start

Train a parametric QFT basis on a target image with Riemannian gradient
descent:

```python
import jax
import jax.numpy as jnp
import pdft

target = jax.random.normal(jax.random.PRNGKey(7), (4, 4)).astype(jnp.complex128)
basis = pdft.QFTBasis(m=2, n=2)

result = pdft.train_basis(
    basis,
    target=target,
    loss=pdft.L1Norm(),
    optimizer=pdft.RiemannianGD(lr=0.01),
    steps=50,
    seed=0,
)
print(result.loss_history[0], "->", result.loss_history[-1])
```

Runnable demos live in [`examples/`](examples/) (each finishes in under
10 seconds):

```bash
python examples/basis_demo.py           # train a QFTBasis, plot the loss
python examples/optimizer_benchmark.py  # GD vs Adam comparison
python examples/mera_demo.py            # MERA basis training
```

## Coherence with the pixel basis

Compression cares only how few coefficients a basis needs. Anything that
recovers an image from a *subset of its pixels* — inpainting, completion,
compressed sensing — is governed by a second quantity, and it is not sparsity:

    mu(U) = N max_ij |U_ij|^2  in [1, N]

`mu = 1` is maximal incoherence with the pixel basis, the best case for
recovery from pointwise samples; `mu = N` is an atom living on one pixel,
invisible to any sample set that misses it.

Every basis here starts at `mu = 1`, and there is a structural reason it can
stay there: if the only non-diagonal gates are one Hadamard per wire, then
`|U_ij| = N^{-1/2}` for **every** parameter value, so `mu = 1` identically and
`sqrt(N) U` is a complex Hadamard matrix. Training the controlled-phase gates
arbitrarily hard, on any objective, cannot move it. Training the Hadamard /
`U(4)` gates can and does — randomising them on a `(3, 3)` `QFTBasis` reaches
`mu = 24.8` out of 64, and on `RichBasis`, which has no diagonal gates at all,
`mu = 32.5`.

`certify_flat_modulus` answers that before a run rather than measuring it
after, using the same `frozen_indices` that `train_basis_batched` takes:

```python
from pdft.coherence import certify_flat_modulus, coherence, diagonal_tensor_indices

basis = pdft.QFTBasis(m=3, n=3)
coherence(basis)                      # 1.0

cert = certify_flat_modulus(basis)    # nothing frozen
print(cert.holds, cert.reason)        # False: the Hadamards are trainable

frozen = cert.offending_indices       # exactly what must be held fixed
assert certify_flat_modulus(basis, frozen_indices=frozen)
result = pdft.train_basis_batched(basis, frozen_indices=frozen, ...)
```

Freezing gates is a real trade — it removes the freedom that `RichBasis` and
`TEBDBasis` add for compression. The point is that the trade is now visible and
checkable, so it can be made deliberately per task.

## Image inpainting from random pixels (`pdft.completion`)

`pdft.completion` is the library half of
[pdft-completion](https://github.com/zazabap/pdft-completion), the code behind
*Image Inpainting from Random Pixels with a Trainable Quantum Fourier
Transform*. The premise follows from the coherence section above: recovery
from a random subset of pixels is governed by incoherence with the pixel
basis, not by sparsity, so the QFT circuit is trained *through* the recovery
solver — `K` unrolled steps of hard thresholding plus data consistency,
differentiated end to end with a straight-through top-k — while Proposition 1
keeps `mu = 1` at every parameter value. Plain Adam moves on the manifold with
no retraction.

The subpackage carries the circuit in a second representation: the gate
angles, applied directly to the image in `O(N log N)` (no matrix, any register
width, float32 or float64), instead of the core package's tensor lists.
`pdft.completion.bridge` converts between the two exactly. A transform family
is one per-axis operator `apply(x, params, adjoint, axis)`; its 2-D pair, its
solver, the batched solver and its evaluation come from `separable`,
`solver_for`, `batched` and `evaluate_params`, and register widths are read
off array shapes rather than passed around.

```python
import numpy as np
import jax.numpy as jnp
from pdft.completion import psnr, qft_basis_from_angles, reconstruct, theta0, train
from pdft.completion.protocol import train_k

n = 9                                    # 512 x 512 images, float32 sets the working precision
images = np.stack([...]).astype(np.float32)
k = train_k(2**n * 2**n, p=0.10, frac=0.125)

params, history = train(images, k, K=100, p=0.10, steps=200, lr=8e-3)

obs = jnp.asarray(np.random.default_rng(0).random((512, 512)) < 0.10)
x_hat = reconstruct(params["r"], params["c"], test_image * obs, obs, k, K=300)
print(psnr(x_hat, test_image), "dB vs the DFT:",
      psnr(reconstruct(theta0(n), theta0(n), test_image * obs, obs, k, 300), test_image))

basis = qft_basis_from_angles(params["r"], params["c"])   # a pdft.QFTBasis: save it, certify it, draw it
```

| Module | What it holds |
|---|---|
| `transform` | the circuit kernel `apply_gates`, the dense operator `apply_dense`, the DFT anchor `theta0` (`U(theta0) == conj(DFT)`, the QFT sign), and `separable`, the 2-D analysis / synthesis pair of any operator |
| `solver` | `hard_k` / `soft_k` (straight-through), the unrolled IHT scan `iht`, and `solver_for` / `batched`, the jitted and vmapped K-step solver of any operator |
| `unroll` | memory-bounded differentiation: nested rematerialisation turns `O(K N^2)` into `O(sqrt(K) N^2)`, with a planner |
| `training` | `minibatches` (the one batch/mask schedule every trainer draws from), `task_loss` of any solver, `mu_monitor`, `adam_loop` |
| `adam` | plain Adam written out to mirror `optax.adam` (bit for bit for real parameters under jit; this package does not depend on optax) |
| `coherence` | `mu` for closures and matrices, and `certify_flat_modulus` over sampled parameters |
| `metrics`, `protocol` | PSNR / SSIM / MS-SSIM, and the paper's Table I protocol as data (`heldout_mask`, `budget_k`, `table1_scores`, `evaluate_params`, …) |
| `data` | Kodak and DIV2K splits from an explicit data directory (needs `pillow`) |
| `bridge` | `qft_basis_from_angles`, `qft_basis_from_general` and their inverses |
| `families/` | one operator each: `phases` (the paper's own family: `apply_u`, `train`, `reconstruct`), `general` (QFT + diagonals, QFT + rotations with a Cayley step), `shared` (phases tied by gate distance, so one fit transfers across resolutions), `butterfly` (Dao et al.), `riemannian` (a free unitary on U(N)), `transform_learning` (a separable orthonormal pair fitted for sparsity) |
| `baselines/` | per-image methods with no trained basis: fixed bases (DCT-II / DFT / wavelets), nuclear-norm completion (SVP, APG), the coarse-to-fine quantized tensor train |

`pip install "pdft[completion]"` adds the optional backends (`pillow` for
image loading, `PyWavelets` for the wavelet baselines, `scipy` for its DCT).
The circuit, solver, trainer and metrics need none of them. The experiment
scripts, data fetchers, figures and result files stay in the paper's
repository; `examples/completion_demo.py` is a self-contained run.

## Background

For the theory, see the paper:
- [Fast Trainable Multilinear Bases for Image Compression](https://arxiv.org/abs/2608.00053) (arXiv:2608.00053)

and the upstream notes:
- [`note/stepbystep.pdf`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/stepbystep.pdf)
- [`note/main.pdf`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/main.pdf)

## Citation

If you use this package in your research, please cite:

```bibtex
@misc{an2026fast,
  title         = {Fast Trainable Multilinear Bases for Image Compression},
  author        = {An, Shiwen and Ni, Zhongyi and Zhou, Huanhai and Liu, Jin-Guo},
  year          = {2026},
  eprint        = {2608.00053},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2608.00053},
}
```

## License

MIT. See [LICENSE](LICENSE). This project is a derivative port of
ParametricDFT.jl (Copyright © 2025 nzy1997, MIT).
