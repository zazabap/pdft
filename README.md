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

## Faster arithmetic

By default a circuit is applied gate by gate, each gate reading and writing
the whole image, with arithmetic that matches the Julia goldens within the
parity tolerances. Two opt-in options apply the same operator with less work
and change only the low bits:

- `merge_diagonals` multiplies each run of consecutive controlled-phase
  gates into one diagonal over the wires the run touches, and applies it to
  the image once. A QFT register of `w` qubits then makes `2w - 1` passes
  over the image instead of `w(w+1)/2`, and the gradient reduces over the
  image once per run instead of once per gate.
- `slices` applies each one-qubit gate as four multiplications by a scalar
  of the two slices of its axis, instead of a contraction.

Use the two together. On a GPU the one-qubit contractions take most of the
time until `slices` removes them; the merge then makes a gradient another
2.5 to 2.9 times faster.

Set them on a basis, for instance one you saved, and its transforms and
every trainer use them:

```python
from dataclasses import replace

import pdft
from pdft.io import load_basis

basis = load_basis("basis.json")
fast = pdft.QFTBasis(
    basis.m, basis.n, tensors=basis.tensors,
    code=replace(basis.code, slices=True, merge_diagonals=True),
)
```

Batched gradients at 256x256 in double precision, both options against the
default (16 images on a GPU, 4 to 16 on a CPU):

| Circuit | GPU | CPU |
|---|---|---|
| QFT, EntangledQFT, TEBD and MERA (`"cp"`) | 11 to 16 times faster | 1.7 to 2 times |
| QFT blocked into 32x32 | 10 times | 1.5 times |
| DCT4, Rich, RealRich, TEBD and MERA (`"u4"`) | 1.3 to 2.5 times | 1.1 to 1.2 times |

The last row never has two controlled phases in a row to merge: its dense
two-qubit gates are contracted either way, and only its one-qubit gates
gain. The gain grows with the image (for a blocked basis, with the block);
at 16x16 on a CPU it is about even. A batched Adam step of
`train_basis_batched` on 256x256 QFT images gains 10 times on a GPU and up
to 1.7 times on a CPU. `train_basis`, most of whose step runs outside the
circuit, gains less: on a GPU 1.2 to 1.5 times at 256x256 and about 5 at
1024x1024.

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
