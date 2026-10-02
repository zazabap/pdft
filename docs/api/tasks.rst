Tasks
=====

.. currentmodule:: pdft.tasks

What a basis is used for, one module per task. A task calls only ``forward_transform`` and ``inverse_transform``, so it runs on every basis. Compression keeps the largest coefficients of an image and rebuilds it from them (upstream's ``compression.jl``, with its JSON form).

.. automodule:: pdft.tasks
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   compress
   compress_with_k
   recover
   CompressedImage
   compression_stats
   save_compressed
   load_compressed
   compressed_to_dict
   dict_to_compressed
