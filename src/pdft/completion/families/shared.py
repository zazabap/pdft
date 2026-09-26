"""Resolution-transferable parameterisation: phases shared by gate distance.

Model B ties a basis to one resolution: 2n(n-1) phases per axis, one set per
gate, and a basis fitted at 512^2 (n = 9, 36 gates) says nothing about the 66
gates of a 4096^2 image. The DFT has no such problem --- its angle
``2 pi / 2^(p-q+1)`` depends on ``(p, q)`` only through the distance
``d = p - q``, which is why one rule generates the circuit at every n. Sharing
the learned phases the same way, ``phi_pq = psi_d``, keeps that: the family
still contains the DFT, still has mu == 1 (nothing touches the Hadamards), and
has 4(n-1) parameters per axis. ``extend`` carries a fit up to a wider register,
the longer-range gates it never saw staying at their textbook value.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..transform import gate_pairs, hadamards


def dist_index(n: int) -> jnp.ndarray:
    """Gate i of gate_pairs(n) draws its phases from psi[dist_index[i]]."""
    return jnp.asarray([p - q - 1 for p, q in gate_pairs(n)], dtype=jnp.int32)


def init_shared(n: int) -> jnp.ndarray:
    """psi at the DFT: (n-1, 4), row d-1 is (0, 0, 0, 2 pi / 2^(d+1))."""
    psi = np.zeros((n - 1, 4))
    psi[:, 3] = [2.0 * np.pi / 2 ** (d + 2) for d in range(n - 1)]
    return jnp.asarray(psi)


def expand(psi: jnp.ndarray, n: int) -> dict:
    """psi -> the ``{"g", "phi"}`` dict the circuit kernel consumes."""
    return {"g": hadamards(n), "phi": psi[dist_index(n)]}


def extend(psi: jnp.ndarray, n_new: int) -> jnp.ndarray:
    """Carry a psi fitted at n_train to n_new, the DFT for the distances it lacks."""
    base = init_shared(n_new)
    return (
        psi[: base.shape[0]] if psi.shape[0] > base.shape[0] else base.at[: psi.shape[0]].set(psi)
    )


def n_shared(n: int) -> int:
    return 4 * (n - 1)
