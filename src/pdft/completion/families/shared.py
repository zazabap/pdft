"""Phases shared by gate distance: the resolution-transferable parameterisation.

Model B ties a basis to one resolution: one set of phases per gate, so a basis
fitted at ``512^2`` (36 gates) says nothing about the 66 gates of a ``4096^2``
image. The DFT has no such problem, since its angle ``2 pi / 2^(p-q+1)``
depends on the pair only through the distance ``d = p - q``, which is why one
rule generates the circuit at every width. Sharing the learned phases the same
way, ``phi_pq = psi_d``, keeps that: the family still contains the DFT, still
has ``mu == 1`` (nothing touches the Hadamards), and has ``4(n-1)`` parameters
per axis. ``extend`` carries a fit to a wider register, the longer-range gates
it never saw staying at their textbook value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..transform import gate_pairs, hadamards

Array = jax.Array


def dist_index(n: int) -> Array:
    """Gate ``i`` of ``gate_pairs(n)`` draws its phases from ``psi[dist_index[i]]``."""
    return jnp.asarray([p - q - 1 for p, q in gate_pairs(n)], dtype=jnp.int32)


def init_shared(n: int) -> Array:
    """``psi`` at the DFT: ``(n-1, 4)``, row ``d-1`` is ``(0, 0, 0, 2 pi / 2^(d+1))``."""
    psi = np.zeros((n - 1, 4))
    psi[:, 3] = [2.0 * np.pi / 2 ** (d + 2) for d in range(n - 1)]
    return jnp.asarray(psi)


def expand(psi: Array, n: int) -> dict:
    """``psi`` as the ``{"g", "phi"}`` gate dict the circuit kernel consumes."""
    return {"g": hadamards(n), "phi": psi[dist_index(n)]}


def extend(psi: Array, n_new: int) -> Array:
    """Carry a ``psi`` fitted at one width to ``n_new``, the DFT for the distances it lacks."""
    base = init_shared(n_new)
    return (
        psi[: base.shape[0]] if psi.shape[0] > base.shape[0] else base.at[: psi.shape[0]].set(psi)
    )


def n_shared(n: int) -> int:
    """Parameters per axis, ``4(n-1)``."""
    return 4 * (n - 1)
