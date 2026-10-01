Serialization and compression
=============================

.. currentmodule:: pdft.io

JSON serialization of a trained ``QFTBasis`` (the Julia package's schema, tensor order and hash, so a basis saved by Julia loads here) and top-k lossy compression of images in a learned basis.

.. automodule:: pdft.io
   :no-members:

.. rubric:: Contents

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
