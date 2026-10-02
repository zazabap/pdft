"""Sparse-basis subpackage.

Every name is importable from here. Where things live:

- ``bases.core`` — what every basis shares: CircuitBasis, the transforms,
  bases_allclose, with_tensors and the parameter views (ParameterView).
- ``bases.base`` — the QFT, EntangledQFT, TEBD, MERA and DCT4 classes.
- ``bases.circuit`` — one module per circuit family with its gate emitter
  (``<family>_gates``); the Rich and RealRich classes; freeze_as_blocked.
- ``bases.block`` — BlockedBasis, an inner basis tiled over image blocks.
"""

from .base import DCT4Basis, EntangledQFTBasis, MERABasis, QFTBasis, TEBDBasis
from .block import BlockedBasis
from .circuit import (
    RealRichBasis,
    RichBasis,
    dct4_code,
    entangled_qft_code,
    fit_to_dct,
    freeze_as_blocked,
    ft_mat,
    ift_mat,
    mera_code,
    qft_code,
    tebd_code,
)
from .core import (
    CP_DIAGONALS,
    CP_PHASES,
    TENSORS,
    AbstractSparseBasis,
    CircuitBasis,
    ParameterView,
    bases_allclose,
    cp_diagonals,
    cp_phases,
    program_of,
    with_cp_diagonals,
    with_cp_phases,
    with_tensors,
)

__all__ = [
    "CP_DIAGONALS",
    "CP_PHASES",
    "TENSORS",
    "AbstractSparseBasis",
    "BlockedBasis",
    "CircuitBasis",
    "DCT4Basis",
    "EntangledQFTBasis",
    "MERABasis",
    "ParameterView",
    "QFTBasis",
    "RealRichBasis",
    "RichBasis",
    "TEBDBasis",
    "bases_allclose",
    "cp_diagonals",
    "cp_phases",
    "dct4_code",
    "entangled_qft_code",
    "fit_to_dct",
    "freeze_as_blocked",
    "ft_mat",
    "ift_mat",
    "mera_code",
    "program_of",
    "qft_code",
    "tebd_code",
    "with_cp_diagonals",
    "with_cp_phases",
    "with_tensors",
]
