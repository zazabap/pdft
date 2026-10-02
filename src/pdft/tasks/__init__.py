"""What a basis is used for: one module per task.

A task takes a basis and data and transforms through ``forward_transform``
and ``inverse_transform``, never through the circuit, so it runs on every
basis in the package.

- ``tasks.compression`` — keep the largest coefficients of an image and
  rebuild it from them. Mirror of upstream ``src/compression.jl``.
- ``tasks.completion`` — fill in an image from the pixels that were observed,
  and the objective for training a basis to do so. Not in upstream.
"""

from .completion import complete, completion_loss
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
    "complete",
    "completion_loss",
    "compress",
    "compress_with_k",
    "compressed_to_dict",
    "compression_stats",
    "dict_to_compressed",
    "load_compressed",
    "recover",
    "save_compressed",
]
