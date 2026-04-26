"""Serialization (JSON) and lossy compression of trained bases."""

from .compression import (
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
from .serialize import (
    basis_hash,
    basis_to_dict,
    dict_to_basis,
    load_basis,
    save_basis,
)

__all__ = [
    "CompressedImage",
    "basis_hash",
    "basis_to_dict",
    "compress",
    "compress_with_k",
    "compressed_to_dict",
    "compression_stats",
    "dict_to_basis",
    "dict_to_compressed",
    "load_basis",
    "load_compressed",
    "recover",
    "save_basis",
    "save_compressed",
]
