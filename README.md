# pdft

A Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl):
learning parametric quantum Fourier transforms via manifold optimization. The
package implements a variational approach that approximates the Discrete
Fourier Transform (DFT) with parameterized quantum circuits.

> Status: early scaffold. The Julia package is the reference implementation;
> this repository will grow the Python equivalent incrementally.

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

_(coming soon — mirrors the `make example` demo in the upstream Julia
package)_

## Background

See the upstream notes for the theory:
- [`note/stepbystep.pdf`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/stepbystep.pdf)
- [`note/main.pdf`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/main.pdf)

## License

MIT. See [LICENSE](LICENSE). This project is a derivative port of
ParametricDFT.jl (Copyright © 2025 nzy1997, MIT).
