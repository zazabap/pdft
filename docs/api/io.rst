Serialization and compression
=============================

.. currentmodule:: pdft.io

JSON serialization of trained bases (byte-compatible with the Julia package) and top-k lossy compression of images in a learned basis.

.. autosummary::
   :toctree: generated
   :nosignatures:

   save_basis
   load_basis
   basis_to_dict
   dict_to_basis
   basis_hash
   format_float_julia_like
   compress
   compress_with_k
   recover
   CompressedImage
   compression_stats
   save_compressed
   load_compressed
   compressed_to_dict
   dict_to_compressed
