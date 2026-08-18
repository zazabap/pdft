# pdft

[![arXiv](https://img.shields.io/badge/arXiv-2608.00053-b31b1b.svg)](https://arxiv.org/abs/2608.00053)

A Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl):
learning parametric quantum Fourier transforms via manifold optimization. The
package implements a variational approach that approximates the Discrete
Fourier Transform (DFT) with parameterized quantum circuits.

This is the reference implementation accompanying the paper
[*Fast Trainable Multilinear Bases for Image Compression*](https://arxiv.org/abs/2608.00053)
(An, Ni, Zhou, Liu, 2026).

> Status: feature-complete port. All bases (QFT, entangled QFT, TEBD, MERA,
> Rich/RealRich, DCT-IV, blocked), both Riemannian optimizers (GD + Adam),
> training, JSON/compression I/O, and visualization are implemented, with
> parity against the Julia reference verified by committed goldens.

## Installation

Once published on PyPI:

```bash
pip install pdft
```

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
