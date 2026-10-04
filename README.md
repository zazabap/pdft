<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/zazabap/pdft/main/docs/_static/logo-dark.svg">
    <img alt="pdft" src="https://raw.githubusercontent.com/zazabap/pdft/main/docs/_static/logo-light.svg" width="340">
  </picture>
</p>

# pdft

[![compression: arXiv 2608.00053](https://img.shields.io/badge/compression-arXiv%202608.00053-b31b1b.svg)](https://arxiv.org/abs/2608.00053)
[![inpainting: arXiv 2609.17298](https://img.shields.io/badge/inpainting-arXiv%202609.17298-b31b1b.svg)](https://arxiv.org/abs/2609.17298)
[![Docs](https://img.shields.io/badge/docs-zazabap.github.io%2Fpdft-blue.svg)](https://zazabap.github.io/pdft/)

Trainable quantum-circuit transforms for images, in JAX. A circuit of
Hadamard and controlled-phase gates starts as the quantum Fourier transform;
its gates are trained so that images become sparse in the basis (compression),
or so that a sparse-recovery solver fills them in from a fraction of their
pixels (inpainting). Training moves the gates on their unitary manifolds, or
the phases alone as free numbers, and the circuit's structure keeps its
coherence with the pixel basis at the minimum throughout.

`pdft` is a Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl)
and the reference implementation of the two papers above. The API reference
and example gallery are at [zazabap.github.io/pdft](https://zazabap.github.io/pdft/).

> Parity with the Julia reference is verified by committed goldens: for
> `QFTBasis`, the transform, the losses and top-k truncation, the manifold
> operations, both optimizers' trajectories, the JSON format and compression;
> for entangled QFT, TEBD and MERA, the forward transform at default options,
> and for entangled QFT the phase extraction. Rich/RealRich, DCT-IV, the
> blocked bases and everything under `pdft.tasks.completion` have no Julia
> counterpart: they are covered by property tests, and completion by a golden
> from the inpainting paper's own code. Not everything upstream exports is
> ported: JSON for bases other than `QFTBasis`, the `:middle` entangle
> position, the loss-history files, device transfer and some of the plots.

## Installation

From PyPI (Python 3.11+):

```bash
pip install "pdft>=0.3.0"
```

> **Note:** 0.3.0 is the first release with the completion task, the step
> trainer and the parameter views, and it moves `compress` / `recover` from
> `pdft.io` to `pdft.tasks`. The older `pdft==0.2.2` wheel predates
> `DCT4Basis` and the `parametrization="u4"` option of `TEBDBasis` /
> `MERABasis`.

From source:

```bash
git clone https://github.com/zazabap/pdft.git
cd pdft
pip install -e ".[dev]"
```

## Quick start

Train a parametric QFT basis on a target image with Riemannian gradient
descent, the compression objective:

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

Fill in an image from a tenth of its pixels, then train the phases of the
circuit through that solver (the inpainting paper's Model B):

```python
from pdft.bases import cp_diagonals_view
from pdft.circuit import bit_reverse
from pdft.tasks import complete, completion_loss

# image: a (64, 64) array in [0, 1]; mask: bool, True where a pixel was observed;
# images: a stack of training images. A QFTBasis works on the bit-reversed
# image, and a mask lives on the pixels, hence the reversals.
basis = pdft.QFTBasis(m=6, n=6)
filled = bit_reverse(complete(basis, bit_reverse(image * mask), bit_reverse(mask), k=61, steps=60))

result = pdft.train_basis_steps(
    basis,
    dataset=images,                                  # in the image's own frame
    objective=completion_loss(k=61, steps=10),
    view=cp_diagonals_view,                          # the four phases of every gate, plain Adam
    optimizer=pdft.RiemannianAdam(lr=0.02),
    steps=60, rate=0.10, batch_size=2, seed=0,
    frame=bit_reverse,
)
```

Runnable demos live in [`examples/`](examples/) (each takes a few seconds):

```bash
python examples/basis_demo.py           # train a QFTBasis, plot the loss
python examples/optimizer_benchmark.py  # GD vs Adam comparison
python examples/mera_demo.py            # MERA basis training
python examples/completion_demo.py      # fill in an image, train through the solver
```

## Coherence with the pixel basis

Compression cares only how few coefficients a basis needs. Recovering an
image from a *subset of its pixels* is governed by a second quantity, the
coherence `mu(U) = N max_ij |U_ij|^2` in `[1, N]`: `mu = 1` is maximal
incoherence with the pixel basis, `mu = N` an atom living on one pixel.

The QFT-family bases start at `mu = 1`, and there is a structural reason it
can stay there: if the only non-diagonal gates are one Hadamard per wire,
then `|U_ij| = N^{-1/2}` for every parameter value. Training the
controlled-phase gates, on any objective, cannot move `mu`; training the
Hadamard or `U(4)` gates can and does. `certify_flat_modulus` answers that
before a run, with the same `frozen_indices` the trainers take, and the
phase views (`cp_phases_view`, `cp_diagonals_view`) train nothing else:

```python
from pdft.coherence import certify_flat_modulus

basis = pdft.QFTBasis(m=3, n=3)
cert = certify_flat_modulus(basis)              # False: the Hadamards are trainable
frozen = cert.offending_indices                 # exactly what must be held fixed
assert certify_flat_modulus(basis, frozen_indices=frozen)
result = pdft.train_basis_batched(basis, frozen_indices=frozen, ...)
```

## Background

- [Fast Trainable Multilinear Bases for Image Compression](https://arxiv.org/abs/2608.00053)
  (An, Ni, Zhou, Liu, 2026): the bases, the losses, the Riemannian optimizers.
- [Quantum-Inspired Trainable and Parameter-Efficient Tensor Networks for Image Inpainting](https://arxiv.org/abs/2609.17298)
  (An, Slavakis, 2026): the completion solver, training through it, and the
  coherence argument.
- The upstream notes: [`note/stepbystep.pdf`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/stepbystep.pdf),
  [`note/main.pdf`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/main.pdf).

## License

MIT. See [LICENSE](LICENSE). This project is a derivative port of
ParametricDFT.jl (Copyright © 2025 nzy1997, MIT).
