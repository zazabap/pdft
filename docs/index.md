# pdft

```{rst-class} hero-tagline
```
Trainable quantum-circuit transforms for image compression and inpainting, in JAX.

`pdft` starts from the quantum Fourier transform as a circuit of Hadamard and
controlled-phase gates and trains the gates: on their unitary manifolds so an
image becomes sparse in the basis, or the phases alone, as free numbers,
through a solver that fills an image in from a fraction of its pixels. It is a
faithful Python port of
[ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl), with results
checked against committed Julia goldens, and the reference implementation of
[two papers](paper.md).

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Bases, optimizers, losses, training loops, the compression and completion
tasks, I/O and coherence tools, generated from the source docstrings.
:::

:::{grid-item-card} Example gallery
:link: auto_examples/index
:link-type: doc

Short runnable scripts that train bases, compare optimizers and fill in an
image from a tenth of its pixels, with their figures rendered at build time.
:::

:::{grid-item-card} Papers
:link: paper
:link-type: doc

The two arXiv papers this package accompanies, and which example
demonstrates each.
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
| Training | {func}`~pdft.training.train_basis` (single target), {func}`~pdft.training.train_basis_batched` (multi-image, cosine schedule, early stopping), {func}`~pdft.training.train_basis_steps` (a fresh batch and mask per step, for training through a solver) |
| Parameter views | {data}`~pdft.bases.tensors_view`, {data}`~pdft.bases.cp_phases_view`, {data}`~pdft.bases.cp_diagonals_view`: what a trainer moves |
| Tasks | {mod}`pdft.tasks`: top-k compression ({func}`~pdft.tasks.compress`, {func}`~pdft.tasks.recover`) and completion from observed pixels ({func}`~pdft.tasks.complete`, {func}`~pdft.tasks.completion_loss`) |
| Coherence | {func}`~pdft.coherence.coherence`, {func}`~pdft.coherence.certify_flat_modulus` |
| I/O | {mod}`pdft.io`: JSON serialization compatible with Julia |

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
