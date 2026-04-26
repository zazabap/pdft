"""Sparse-basis subpackage.

Two families:
- bases.circuit  — full circuit topologies (QFT, EntangledQFT, TEBD, MERA),
                   comparable to FFT/DCT.
- bases.block    — parameter-efficient block-structured bases (Blocked,
                   Rich, RealRich) over arbitrary block partitions.

The abstract base class and bases_allclose helper live in bases.base.
"""

from .base import (
    AbstractSparseBasis,
    EntangledQFTBasis,
    MERABasis,
    QFTBasis,
    TEBDBasis,
    bases_allclose,
)
from .block import BlockedBasis, RealRichBasis, RichBasis, fit_to_dct
from .circuit import entangled_qft_code, ft_mat, ift_mat, mera_code, qft_code, tebd_code

__all__ = [
    "AbstractSparseBasis",
    "BlockedBasis",
    "EntangledQFTBasis",
    "MERABasis",
    "QFTBasis",
    "RealRichBasis",
    "RichBasis",
    "TEBDBasis",
    "bases_allclose",
    "entangled_qft_code",
    "fit_to_dct",
    "ft_mat",
    "ift_mat",
    "mera_code",
    "qft_code",
    "tebd_code",
]
