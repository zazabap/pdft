"""pdft: Python port of ParametricDFT.jl (faithful JAX reference).

Importing this package enables JAX's x64 mode globally. This is required
to match Julia's ComplexF64 numerical behavior; without it, parity
tolerances are unreachable. See docs/superpowers/specs/2026-04-24-pdft-migration-design.md
Section 2 and 8.1.

Also enables a persistent JAX compilation cache by default, so the
~20-second JIT-compile cost of the Adam step is paid once per basis
shape across all runs. Override the cache dir with JAX_COMPILATION_CACHE_DIR
or disable entirely by setting PDFT_DISABLE_COMPILE_CACHE=1.

Public API: import from subpackages directly, e.g.

    from pdft.bases.circuit import QFTBasis
    from pdft.training import train_basis, train_basis_batched
    from pdft.optimizers import RiemannianGD, RiemannianAdam
    from pdft.loss import L1Norm, MSELoss
    from pdft.io import save_basis, load_basis, compress

The names re-exported at the package root below are kept only for the
small set most commonly used in interactive sessions and notebooks.
"""

import os as _os
from pathlib import Path as _Path

import jax as _jax

_jax.config.update("jax_enable_x64", True)

if not _os.environ.get("PDFT_DISABLE_COMPILE_CACHE"):
    _cache_dir = _os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not _cache_dir:
        _xdg = _os.environ.get("XDG_CACHE_HOME") or str(_Path.home() / ".cache")
        _cache_dir = str(_Path(_xdg) / "pdft" / "jax-compile-cache")
    try:
        _Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", _cache_dir)
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except OSError:
        pass

__version__ = "0.0.0"
__upstream_ref__ = "nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef"

# Slim public re-export hub: most-used names only. Anything else: import
# from the relevant subpackage directly.
from .bases import (  # noqa: E402
    AbstractSparseBasis,
    BlockedBasis,
    EntangledQFTBasis,
    MERABasis,
    QFTBasis,
    RealRichBasis,
    RichBasis,
    TEBDBasis,
    bases_allclose,
)
from .loss import AbstractLoss, L1Norm, MSELoss, loss_function  # noqa: E402
from .optimizers import RiemannianAdam, RiemannianGD, optimize  # noqa: E402
from .training import TrainingResult, train_basis, train_basis_batched  # noqa: E402

__all__ = [
    "AbstractLoss",
    "AbstractSparseBasis",
    "BlockedBasis",
    "EntangledQFTBasis",
    "L1Norm",
    "MERABasis",
    "MSELoss",
    "QFTBasis",
    "RealRichBasis",
    "RichBasis",
    "RiemannianAdam",
    "RiemannianGD",
    "TEBDBasis",
    "TrainingResult",
    "__upstream_ref__",
    "__version__",
    "bases_allclose",
    "loss_function",
    "optimize",
    "train_basis",
    "train_basis_batched",
]
