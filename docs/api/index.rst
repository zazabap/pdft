API reference
=============

The names used most often are re-exported at the package root, so
``pdft.QFTBasis`` and ``pdft.bases.QFTBasis`` are the same object. Everything
else is imported from its subpackage, for example
``from pdft.io import save_basis``.

.. toctree::
   :caption: Core
   :maxdepth: 1

   bases
   optimizers
   loss
   training
   coherence
   completion
   io

.. toctree::
   :caption: Internals and tools
   :maxdepth: 1

   manifolds
   circuit
   viz
   profiling
