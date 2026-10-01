"""The circuit families: one module each, with the family's gate emitter.

``<family>_gates(m, n, ...)`` emits the gate list a basis class compiles;
``<family>_code`` is upstream's older entry point, the applier and initial
tensors of the same gates. Rich and RealRich (H + two-qubit gates on the QFT
topology) define their basis classes here too; the other classes are in
``pdft.bases.base``.
"""

from .dct4 import dct4_code, dct4_ft_mat, dct4_gates, dct4_ift_mat
from .entangled_qft import entangled_qft_code, entangled_qft_gates
from .freeze import freeze_as_blocked
from .mera import mera_code, mera_gates
from .qft import ft_mat, ift_mat, qft_code, qft_gates
from .real_rich import RealRichBasis, real_rich_gates
from .rich import RichBasis, fit_to_dct, rich_gates
from .tebd import tebd_code, tebd_gates

__all__ = [
    "RealRichBasis",
    "RichBasis",
    "dct4_code",
    "dct4_ft_mat",
    "dct4_gates",
    "dct4_ift_mat",
    "entangled_qft_code",
    "entangled_qft_gates",
    "fit_to_dct",
    "freeze_as_blocked",
    "ft_mat",
    "ift_mat",
    "mera_code",
    "mera_gates",
    "qft_code",
    "qft_gates",
    "real_rich_gates",
    "rich_gates",
    "tebd_code",
    "tebd_gates",
]
