"""pdft: Python port of ParametricDFT.jl (faithful JAX reference).

Importing this package enables JAX's x64 mode globally. This is required
to match Julia's ComplexF64 numerical behavior; without it, parity
tolerances are unreachable. See docs/superpowers/specs/2026-04-24-pdft-migration-design.md
Section 2 and 8.1.
"""
import jax as _jax

_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.0"
__upstream_ref__ = "nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef"

from .basis import AbstractSparseBasis, QFTBasis, bases_allclose  # noqa: E402
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
from .qft import ft_mat, ift_mat, qft_code  # noqa: E402
from .training import TrainingResult, train_basis  # noqa: E402

__all__ = [
    "AbstractLoss",
    "AbstractRiemannianManifold",
    "AbstractSparseBasis",
    "L1Norm",
    "MSELoss",
    "PhaseManifold",
    "QFTBasis",
    "RiemannianAdam",
    "RiemannianGD",
    "TrainingResult",
    "UnitaryManifold",
    "__upstream_ref__",
    "__version__",
    "bases_allclose",
    "basis_hash",
    "basis_to_dict",
    "classify_manifold",
    "dict_to_basis",
    "ft_mat",
    "group_by_manifold",
    "ift_mat",
    "load_basis",
    "loss_function",
    "optimize",
    "qft_code",
    "save_basis",
    "topk_truncate",
    "train_basis",
]
