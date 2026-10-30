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
pip install "pdft>=0.4.0"
```

> **Note:** 0.4.0 turns the faster `slices` and `merge_diagonals`
> arithmetic on by default, so results differ from 0.3.0 by rounding; a
> `CircuitCode` with both off gives 0.3.0's numbers. 0.3.0 is the first
> release with the completion task, the step trainer and the parameter
> views, and it moves `compress` / `recover` from `pdft.io` to
> `pdft.tasks`. The older `pdft==0.2.2` wheel predates
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

A separable circuit (no gate between the two registers: QFT, Rich, RealRich,
DCT4, TEBD, MERA) can also be trained with no gradient steps at all.
`train_basis_exact` alternates the codes `H_k(U x)` with an exact update of
every gate given the others (the polar factor of its environment, or its
phases for a phase tensor), so the sparsity objective never rises; a
`RichBasis` started at the QFT moves away from it, which phase training cannot:

```python
import numpy as np
from pdft.circuit import bit_reverse

result = pdft.train_basis_exact(
    pdft.RichBasis(m=6, n=6),
    dataset=[bit_reverse(x) / np.linalg.norm(x) for x in images],  # each image weighs the same
    k=[205, 410],                                                  # budgets, averaged
    sweeps=20, passes=2,
)
```

`precision="single"` computes everything the size of an image in single
precision, for every basis; the sums over the images and the updates stay in
double. On an RTX 5070, whose double precision is slow, a sweep over four
`2^12 x 2^13` images took 5.6 s instead of 59 s, with the same losses to `1e-6`.

Runnable demos live in [`examples/`](examples/) (each takes a few seconds):

```bash
python examples/basis_demo.py           # train a QFTBasis, plot the loss
python examples/optimizer_benchmark.py  # GD vs Adam comparison
python examples/mera_demo.py            # MERA basis training
python examples/completion_demo.py      # fill in an image, train through the solver
```

## Colour images

A basis acts on one plane. A colour image is handled as a stack of channel
planes, `(3, H, W)`, in a colour space that decorrelates the channels, and
the tasks keep one coefficient budget over all three planes together. That is
the part of JPEG's colour handling worth having here: JPEG converts to YCbCr
and then spends its bits unevenly across the channels. Its other colour step,
halving the chroma resolution, costs quality under a coefficient budget and
is not done.

```python
import numpy as np
from PIL import Image
import pdft
from pdft.color import YCBCR, klt
from pdft.tasks import compress, recover

rgb = np.asarray(Image.open("photo.png").convert("RGB"), dtype=np.float64) / 255  # (256, 256, 3)
basis = pdft.QFTBasis(m=8, n=8)

planes = YCBCR.to_planes(rgb)                # (3, 256, 256): Y, Cb, Cr
kept = compress(basis, planes, ratio=0.9)    # the 10% largest coefficients over all three planes
back = YCBCR.to_image(recover(basis, kept))  # (256, 256, 3)

space = klt(training_images)                 # or the principal colour axes of a dataset, (N, H, W, 3)
```

`complete` and `completion_loss` take the same stacks, with one mask over the
pixels shared by the channels, and `train_basis_steps` takes a dataset of
them and draws its masks that way. Training on channel planes as separate
images (`train_basis_batched`) works as it always did, and one basis trained
so serves all three channels: on DIV2K, separate luma and chroma bases were
never better.

Measured on 100 DIV2K validation images at 256x256 with the DCT-II as the
transform (RGB PSNR; coefficients kept as a share of the 3 N^2 colour
values): a colour space with one shared budget gains 1.3 dB at 1% and 5 dB at
20% over RGB, and the budget split evenly per channel gains nothing. `YCBCR`
is within 0.1 to 0.4 dB of `klt`, which is orthonormal, so the error in its
coefficients is the error in RGB. Colour costs 1.5 to 1.8 times the
coefficients of the luma plane alone at equal PSNR.

## Arithmetic

A circuit is applied to an image one gate at a time. Two options of the
applier cut the work, and both are on by default (in 0.3.0 they were
off):

- `merge_diagonals` multiplies each run of consecutive controlled-phase
  gates into one diagonal over the wires the run touches, and applies it to
  the image once. A QFT register of `w` qubits then makes `2w - 1` passes
  over the image instead of `w(w+1)/2`, and the gradient reduces over the
  image once per run instead of once per gate.
- `slices` applies each one-qubit gate, and the rotation of a controlled
  `R_y` gate, as multiplications by scalars of the two slices of its axis,
  instead of a contraction.

They change how the gates are computed, not what: results agree with the
contraction arithmetic of 0.3.0 to rounding, and the Julia parity tests pass
with them on a CPU and a GPU. To get 0.3.0's numbers bit for bit on a CPU,
turn both off on a basis, for instance one you saved:

```python
from dataclasses import replace

import pdft
from pdft.io import load_basis

basis = load_basis("basis.json")
as_before = pdft.QFTBasis(
    basis.m, basis.n, tensors=basis.tensors,
    code=replace(basis.code, slices=False, merge_diagonals=False),
)
```

Batched gradients at 256x256 in double precision, against the contraction
arithmetic (an RTX 5070 with 16 images, a 6-core Ryzen 5 9600X with 4 to
16; compilation excluded):

| Circuit | GPU | CPU |
|---|---|---|
| QFT, EntangledQFT, TEBD and MERA (`"cp"`) | 11 to 16 times faster | 1.7 to 2 times |
| QFT blocked into 32x32 | 10 times | 1.5 times |
| DCT4 (`parametrization="controlled"`) | 1.9 times | 1.1 times |
| DCT4 (`"o4"`), Rich, RealRich, TEBD and MERA (`"u4"`) | 1.3 to 2.5 times | 1.1 to 1.2 times |

The last two rows have no runs of controlled phases to merge, and their
dense two-qubit gates are contracted either way. Only their one-qubit gates
gain, and the controlled rotations of the controlled DCT4. The gain grows
with the image (for a blocked basis, with the block), and at 16x16 on a CPU
it is about even. In single precision the first row gains 8 to 9 times on
the GPU. The cost is compilation: a 512x512 controlled DCT4 training step
compiles in 29 s instead of 9.

Training gains what the gradient gains: a step of `train_basis_batched`
costs its gradient and about a millisecond more (an Adam step on 16 QFT
images of 256x256 takes 9 ms instead of 94 on the GPU, and 118 instead of
198 on the CPU). On the RTX 5070, 60 epochs
of Adam over 500 images of 512x512 (`MSELoss` keeping 10%, 75 images held
out for validation, the largest batch that fits in 12 GB), estimated from
one timed epoch, take about 2 minutes for a `QFTBasis` instead of 25 (the
options also halve the memory, so its batch is 16 instead of 8), 1.6 hours
for a `DCT4Basis(parametrization="controlled")` instead of 6.5 (at this
size it gains more than in the table), 2.5 hours for the `"o4"` one instead
of 3.5, and 46 minutes for a `RichBasis` instead of 66. `train_basis`, most of whose step runs outside
the circuit, gains less: on a GPU 1.0 to 1.6 times at 256x256 and about 5
at 1024x1024.

The FFT is the baseline. At its initial tensors a `QFTBasis` computes the
same transform as the FFT: `basis.forward_transform(bit_reverse(x))` is
`np.fft.ifft2(x, norm="ortho")`. The FFT is faster at every size. One
transform of a complex128 image on a CPU (Ryzen 5 9600X, 6 cores, JAX
0.10.2), in ms, median of repeated calls with compilation excluded:

| Image | FFT (`jnp.fft.fft2`) | `QFTBasis` | `QFTBasis`, both options off |
|---|---|---|---|
| 64x64 | 0.014 | 0.12 | 0.11 |
| 128x128 | 0.059 | 0.31 | 0.61 |
| 256x256 | 0.11 | 0.75 | 1.4 |
| 512x512 | 0.38 | 3.4 | 6.7 |
| 1024x1024 | 2.6 | 21 | 36 |
| 2048x2048 | 19 | 152 | 267 |

A transform takes 5 to 9 times as long as the FFT, and with both options
off 8 to 18 times. What a trained basis offers over the FFT is a
sparser representation of the images it was trained on, not speed.

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
