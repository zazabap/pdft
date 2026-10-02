"""What a basis is used for: one module per task.

A task is written against ``forward_transform`` and ``inverse_transform``
alone, so it runs on every basis in the package.

- ``tasks.compression`` — keep the largest coefficients of an image and
  rebuild it from them. Mirror of upstream ``src/compression.jl``.
"""

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

__all__ = [
    "CompressedImage",
    "compress",
    "compress_with_k",
    "compressed_to_dict",
    "compression_stats",
    "dict_to_compressed",
    "load_compressed",
    "recover",
    "save_compressed",
]
