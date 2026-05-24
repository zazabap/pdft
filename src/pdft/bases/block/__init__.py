"""Block-structured bases: BlockedBasis.

RichBasis/RealRichBasis/fit_to_dct now live in bases.circuit (they are full
circuits); re-exported here for back-compat.
"""

from ..circuit import RealRichBasis, RichBasis, fit_to_dct
from .block import BlockedBasis

__all__ = [
    "BlockedBasis",
    "RealRichBasis",
    "RichBasis",
    "fit_to_dct",
]
