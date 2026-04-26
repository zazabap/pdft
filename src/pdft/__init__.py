"""pdft: Python port of ParametricDFT.jl (faithful JAX reference).

Importing this package enables JAX's x64 mode globally. This is required
to match Julia's ComplexF64 numerical behavior; without it, parity
tolerances are unreachable. See docs/superpowers/specs/2026-04-24-pdft-migration-design.md
Section 2 and 8.1.

Also enables a persistent JAX compilation cache by default, so the
~20-second JIT-compile cost of the Adam step is paid once per basis
shape across all runs. Override the cache dir with JAX_COMPILATION_CACHE_DIR
or disable entirely by setting PDFT_DISABLE_COMPILE_CACHE=1.
"""

import os as _os
from pathlib import Path as _Path

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Persistent compile cache. Cuts ~20s JIT cost per fresh process. Honors
# JAX_COMPILATION_CACHE_DIR if set, else uses XDG_CACHE_HOME or ~/.cache.
# Opt out with PDFT_DISABLE_COMPILE_CACHE=1.
if not _os.environ.get("PDFT_DISABLE_COMPILE_CACHE"):
    _cache_dir = _os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not _cache_dir:
        _xdg = _os.environ.get("XDG_CACHE_HOME") or str(_Path.home() / ".cache")
        _cache_dir = str(_Path(_xdg) / "pdft" / "jax-compile-cache")
    try:
        _Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", _cache_dir)
        # Cache miss penalty thresholds — cache anything that took >0s to compile,
        # and require at least 1 use of the executable before eviction.
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except OSError:
        # Filesystem read-only or unwritable — silently skip caching.
        pass

__version__ = "0.0.0"
__upstream_ref__ = "nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef"

from .basis import (  # noqa: E402
    AbstractSparseBasis,
    EntangledQFTBasis,
    MERABasis,
    QFTBasis,
    TEBDBasis,
    bases_allclose,
)
from .compression import (  # noqa: E402
    CompressedImage,
    compress,
    compress_with_k,
    compressed_to_dict,
    compression_stats,
    dict_to_compressed,
    load_compressed,
    recover,
    save_compressed,
)
from .entangled_qft import entangled_qft_code  # noqa: E402
from .mera import mera_code  # noqa: E402
from .tebd import tebd_code  # noqa: E402
from .loss import (  # noqa: E402
    AbstractLoss,
    L1Norm,
    MSELoss,
    loss_function,
    topk_truncate,
)
from .manifolds import (  # noqa: E402
    AbstractRiemannianManifold,
    PhaseManifold,
    UnitaryManifold,
    classify_manifold,
    group_by_manifold,
)
from .io_json import (  # noqa: E402
    basis_hash,
    basis_to_dict,
    dict_to_basis,
    load_basis,
    save_basis,
)
from .optimizers import RiemannianAdam, RiemannianGD, optimize  # noqa: E402
from .profiling import ProfileReport, profile_training  # noqa: E402
from .qft import ft_mat, ift_mat, qft_code  # noqa: E402
from .training import TrainingResult, train_basis, train_basis_batched  # noqa: E402

__all__ = [
    "AbstractLoss",
    "AbstractRiemannianManifold",
    "AbstractSparseBasis",
    "CompressedImage",
    "EntangledQFTBasis",
    "L1Norm",
    "MERABasis",
    "MSELoss",
    "PhaseManifold",
    "ProfileReport",
    "QFTBasis",
    "RiemannianAdam",
    "RiemannianGD",
    "TEBDBasis",
    "TrainingResult",
    "UnitaryManifold",
    "__upstream_ref__",
    "__version__",
    "bases_allclose",
    "basis_hash",
    "basis_to_dict",
    "classify_manifold",
    "compress",
    "compress_with_k",
    "compressed_to_dict",
    "compression_stats",
    "dict_to_basis",
    "dict_to_compressed",
    "entangled_qft_code",
    "ft_mat",
    "group_by_manifold",
    "ift_mat",
    "load_basis",
    "load_compressed",
    "loss_function",
    "mera_code",
    "optimize",
    "profile_training",
    "qft_code",
    "recover",
    "save_basis",
    "save_compressed",
    "tebd_code",
    "topk_truncate",
    "train_basis",
    "train_basis_batched",
]
