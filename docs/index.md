# pdft

```{rst-class} hero-tagline
```
Learning parametric quantum Fourier transforms via manifold optimization, in JAX.

`pdft` approximates the discrete Fourier transform with a trainable,
parameterized quantum circuit and optimizes it on the unitary manifold so an
image becomes sparse in the learned basis. It is a faithful Python port of
[ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl), with results
checked against committed Julia goldens, and the reference implementation for
the paper [*Fast Trainable Multilinear Bases for Image Compression*](paper.md).

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Bases, Riemannian optimizers, losses, training loops, I/O and coherence
tools, generated from the source docstrings.
:::

:::{grid-item-card} Example gallery
:link: auto_examples/index
:link-type: doc

Short runnable scripts that train bases and compare optimizers, with their
loss curves rendered at build time.
:::

:::{grid-item-card} Paper & citation
:link: paper
:link-type: doc

The arXiv paper this package accompanies, and the BibTeX entry to cite it.
:::
::::

## Installation

```{include} ../README.md
:start-after: "## Installation"
:end-before: "## Quick start"
```

Optional extras: `pdft[plot]` adds matplotlib for the plotting helpers in
{mod}`pdft.viz`, and `pdft[gpu]` installs a CUDA 12 build of JAX.

```{note}
Importing `pdft` turns on JAX's 64-bit mode for the whole process. Julia
parity depends on `complex128` arithmetic, so import `pdft` before you create
any JAX arrays.
```

## Quick start

```{include} ../README.md
:start-after: "## Quick start"
:end-before: "Runnable demos"
```

## What's in the package

| Area | Contents |
| --- | --- |
| Circuit bases | {class}`~pdft.bases.QFTBasis`, {class}`~pdft.bases.EntangledQFTBasis`, {class}`~pdft.bases.TEBDBasis`, {class}`~pdft.bases.MERABasis`, {class}`~pdft.bases.RichBasis`, {class}`~pdft.bases.RealRichBasis`, {class}`~pdft.bases.DCT4Basis` |
| Block bases | {class}`~pdft.bases.BlockedBasis`, {func}`~pdft.bases.freeze_as_blocked` |
| Optimizers | {class}`~pdft.optimizers.RiemannianGD` (Armijo line search), {class}`~pdft.optimizers.RiemannianAdam` |
| Losses | {class}`~pdft.loss.L1Norm`, {class}`~pdft.loss.MSELoss` with top-k truncation |
| Training | {func}`~pdft.training.train_basis` (single target), {func}`~pdft.training.train_basis_batched` (multi-image, cosine schedule, early stopping) |
| Coherence | {func}`~pdft.coherence.coherence`, {func}`~pdft.coherence.certify_flat_modulus` |
| I/O | {mod}`pdft.io`: JSON serialization compatible with Julia, and top-k compression |

```{toctree}
:hidden:

api/index
auto_examples/index
paper
```

```{toctree}
:hidden:
:caption: Project

GitHub <https://github.com/zazabap/pdft>
PyPI <https://pypi.org/project/pdft/>
ParametricDFT.jl <https://github.com/nzy1997/ParametricDFT.jl>
```
