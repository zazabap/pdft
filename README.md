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
