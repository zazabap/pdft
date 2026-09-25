"""Resolution-transferable parameterisation: phases shared by gate distance.

Model B gives every two-qubit gate (p, q) its own four diagonal phases, so the
basis has 2n(n-1) parameters per axis and is tied to one resolution: a basis
fitted at 512^2 (n = 9, 36 gates) has nothing to say about the 66 gates of a
4096^2 image.

The DFT itself does not have that problem. Its angle

    theta0_pq = 2 pi / 2^(p - q + 1)

depends on p and q only through the distance d = p - q, which is why one rule
generates the circuit at every n. Sharing the learned phases the same way,

    phi_pq = psi_d,      d = p - q in {1, ..., n-1},

keeps that property: the family still contains the DFT exactly, still has mu
identically 1 (nothing here touches the Hadamards), and now has 4(n-1)
parameters per axis --- 32 at n = 9 instead of 144.

A basis fitted at n_train then extends to any n > n_train by keeping the fitted
psi_d for the distances that were seen and falling back to the DFT value for the
longer-range gates that only exist at the larger size (extend). Those gates
carry the finest frequency splittings, which the small image does not contain,
so leaving them at their textbook value is the honest default rather than a
convenience.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..transform import gate_pairs
from .general import _H2


def dist_index(n: int) -> jnp.ndarray:
    """Gate i of gate_pairs(n) draws its phases from psi[dist_index[i]]."""
    return jnp.asarray([p - q - 1 for p, q in gate_pairs(n)], dtype=jnp.int32)


def init_shared(n: int) -> jnp.ndarray:
    """psi at the DFT: (n-1, 4), row d-1 is (0, 0, 0, 2 pi / 2^(d+1))."""
    psi = np.zeros((n - 1, 4))
    psi[:, 3] = [2.0 * np.pi / 2 ** (d + 2) for d in range(n - 1)]
    return jnp.asarray(psi)


def expand(psi: jnp.ndarray, n: int) -> dict:
    """psi -> the per-gate parameter dict that pdft.completion.families.general consumes."""
    return {"g": jnp.broadcast_to(_H2, (n, 2, 2)), "phi": psi[dist_index(n)]}


def extend(psi: jnp.ndarray, n_new: int) -> jnp.ndarray:
    """Carry a psi fitted at n_train up to n_new > n_train, DFT for the rest."""
    base = init_shared(n_new)
    d = psi.shape[0]
    if d > base.shape[0]:
        return psi[: base.shape[0]]
    return base.at[:d].set(psi)


def n_shared(n: int) -> int:
    return 4 * (n - 1)
