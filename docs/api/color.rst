Colour
======

.. currentmodule:: pdft.color

Colour images as a stack of channel planes. A basis acts on one plane; the tasks and the step trainer take the stack ``(C, H, W)`` and keep one coefficient budget over all of it. A :class:`ColorSpace` is the invertible change of channels applied at every pixel that makes the stack from an ``(..., H, W, C)`` image and puts it back: :data:`YCBCR`, the luma and chroma differences of JPEG and video, or :func:`klt`, the principal axes of a dataset's pixel colours.

.. automodule:: pdft.color
   :no-members:

.. rubric:: Contents

.. autosummary::
   :toctree: generated
   :nosignatures:

   ColorSpace
   YCBCR
   klt
