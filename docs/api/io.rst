Serialization
=============

.. currentmodule:: pdft.io

JSON serialization of a trained ``QFTBasis`` (the Julia package's schema, tensor order and hash, so a basis saved by Julia loads here). Compression of an image in a basis, with its JSON form, is in :mod:`pdft.tasks`.

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
