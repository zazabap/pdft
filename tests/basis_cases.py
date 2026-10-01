"""One registry of basis configurations for the tests that run over every basis.

One case per gate kind and parametrization the builder supports, at small
sizes. Rectangular where the basis allows it, so a swapped register cannot go
unnoticed.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

import pdft
from pdft.bases import with_tensors

BASES: dict[str, Callable[[], object]] = {
    "qft_2x2": lambda: pdft.QFTBasis(m=2, n=2),
    "qft_3x2": lambda: pdft.QFTBasis(m=3, n=2),
    "entangled_3x2": lambda: pdft.EntangledQFTBasis(m=3, n=2, seed=1),
    "tebd_cp_3x2": lambda: pdft.TEBDBasis(m=3, n=2, seed=1),
    "tebd_u4_2x2": lambda: pdft.TEBDBasis(m=2, n=2, seed=1, parametrization="u4"),
    "mera_cp_4x2": lambda: pdft.MERABasis(m=4, n=2, seed=1),
    "mera_u4_2x2": lambda: pdft.MERABasis(m=2, n=2, seed=1, parametrization="u4"),
    "dct4_o4_3x2": lambda: pdft.DCT4Basis(m=3, n=2),
    "dct4_controlled_3x2": lambda: pdft.DCT4Basis(m=3, n=2, parametrization="controlled"),
    "rich_3x2": lambda: pdft.RichBasis(m=3, n=2),
    "real_rich_2x2": lambda: pdft.RealRichBasis(m=2, n=2),
    "blocked_qft_2x2_in_3x3": lambda: pdft.BlockedBasis(pdft.QFTBasis(m=2, n=2), 1, 1),
}


def case_rng(case: str) -> np.random.Generator:
    """A generator seeded by the case's name, so each case has its own fixed inputs."""
    return np.random.default_rng(zlib.crc32(case.encode()))


def complex_normal(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def generic(basis, rng: np.random.Generator):
    """The basis at tensors with no symmetry left.

    At initialisation most gates are symmetric or diagonal, which hides a
    swapped leg or a transposed gate. Every entry is scaled by its own complex
    factor; the applier is linear in each tensor, so unitarity is not needed to
    pin what it computes.
    """
    return with_tensors(
        basis, [t * jnp.asarray(1 + 0.2 * complex_normal(rng, t.shape)) for t in basis.tensors]
    )
