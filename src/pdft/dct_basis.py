"""DCTBasis (Approach B): single-matrix orthogonal basis, initialised at DCT.

Motivation: Approach A (`RealRichBasis`) constrains to a 21-dim sub-O(8)
parametric family — strictly smaller than dim O(8) = 28, so DCT may or
may not be in the family. To answer "is DCT itself the optimum, or is
there something better?" cleanly, we need a parametric family that
*provably* contains DCT. The smallest such family is O(8) itself
(= 28 free real params, well below SU(8)'s 63).

`DCTBasis` parametrises each 1D transform as a SINGLE 8×8 real-orthogonal
matrix on the UnitaryManifold (constrained to real-valued by construction
and preserved by Cayley retraction). At initialisation, the matrix IS the
canonical 8×8 DCT-II. Adam refines it on natural-image MSE:

  - if Adam moves AWAY from DCT and lands at a better-PSNR basis, we have
    found a basis genuinely better than DCT for natural images;
  - if Adam stays near DCT, that confirms DCT is approximately a local
    optimum of the natural-image-MSE landscape over O(8).

Naming: this is sometimes called the "Cooley–Tukey-style 1D DCT macro-gate"
in the sense that the entire 1D DCT is one "gate" of the circuit (rather
than being factored into 2-qubit primitives à la Loeffler 1989). The
fully-factorised Loeffler gate sequence is left as future work; the
macro-gate version is sufficient to test "is DCT optimal?".

Parameter count (m=n=3, 8×8 block):
    2 × 28 = 56  (one O(8) per dim, free dim = n(n−1)/2 = 28)
Strictly below SU(8)'s 63 → the parameter budget is meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

Array = jax.Array


def _dct_matrix_real(n: int) -> Array:
    """Orthonormal 1D DCT-II matrix of size n × n, real-valued, complex128."""
    k = np.arange(n).reshape(-1, 1)
    j = np.arange(n)
    M = np.cos(np.pi * (2 * j + 1) * k / (2 * n))
    M[0, :] *= 1.0 / np.sqrt(n)
    M[1:, :] *= np.sqrt(2.0 / n)
    return jnp.asarray(M, dtype=jnp.complex128)


def _macro_einsum_codes(m: int, n: int):
    """Build forward and inverse einsum closures for a 2D macro-gate basis.

    The forward op is `out = D_row @ pic @ D_col.T` where D_row is the
    learnable row-direction 1D unitary (shape 2^m × 2^m) and D_col is the
    column-direction one (shape 2^n × 2^n).

    These closures match the (`*tensors, pic_reshaped`) calling convention
    used by `loss._apply_circuit`, even though there's only one "gate" per
    dimension.
    """
    M, N = 2**m, 2**n

    def forward(D_row: Array, D_col: Array, pic_flat: Array) -> Array:
        # pic_flat is reshaped to (2,)^(m+n); collapse to (M, N) for the matmul.
        pic = pic_flat.reshape(M, N)
        out = D_row @ pic @ jnp.conj(D_col).T  # standard 2D transform
        return out.reshape((2,) * (m + n))

    def inverse(D_row: Array, D_col: Array, pic_flat: Array) -> Array:
        pic = pic_flat.reshape(M, N)
        # Inverse for orthogonal: D^T (or D^H for complex). conj() is applied
        # by the caller (loss._apply_circuit passes conj(tensors)), so here
        # we just transpose.
        out = jnp.conj(D_row).T @ pic @ D_col
        return out.reshape((2,) * (m + n))

    return forward, inverse


@dataclass
class DCTBasis:
    """1D-DCT macro-gate basis (Approach B).

    Stores two unitary matrices (one per dimension), each 2^m × 2^m or
    2^n × 2^n, classified as `UnitaryManifold(d=2^m)` etc. Initialised at
    canonical DCT-II by default. The `tensors` list is `[D_row, D_col]`.

    Parameter count: 2 × O(8) = 56 (for m=n=3). Below SU(8)'s 63.

    Pytree contract:
        leaves   = tensors                                (one list)
        aux data = (m, n, len(tensors), code, inv_code)
    """

    m: int
    n: int
    tensors: list[Array]
    code: object = field(compare=False, repr=False)
    inv_code: object = field(compare=False, repr=False)

    def __init__(
        self,
        m: int,
        n: int,
        init_dct: bool = True,
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        if tensors is None:
            if init_dct:
                D_row = _dct_matrix_real(2**m)
                D_col = _dct_matrix_real(2**n)
            else:
                D_row = jnp.eye(2**m, dtype=jnp.complex128)
                D_col = jnp.eye(2**n, dtype=jnp.complex128)
            tensors = [D_row, D_col]
        fwd, inv = _macro_einsum_codes(m, n)
        self.code = code if code is not None else fwd
        self.inv_code = inv_code if inv_code is not None else inv
        self.tensors = list(tensors)

    @property
    def inv_tensors(self) -> list[Array]:
        return self.tensors

    @property
    def image_size(self) -> tuple[int, int]:
        return (2**self.m, 2**self.n)

    @property
    def num_parameters(self) -> int:
        return sum(int(t.size) for t in self.tensors)

    def forward_transform(self, pic: Array) -> Array:
        from .loss import _apply_circuit

        return _apply_circuit(self.tensors, self.code, self.m, self.n, pic)

    def inverse_transform(self, pic: Array) -> Array:
        from .loss import _apply_circuit

        return _apply_circuit(
            [jnp.conj(t) for t in self.tensors],
            self.inv_code,
            self.m,
            self.n,
            pic,
        )


def _dctbasis_flatten(b: DCTBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _dctbasis_unflatten(aux, leaves) -> DCTBasis:
    m, n, n_fwd, code, inv_code = aux
    assert len(leaves) == n_fwd
    return DCTBasis(m=m, n=n, tensors=list(leaves), code=code, inv_code=inv_code)


tree_util.register_pytree_node(DCTBasis, _dctbasis_flatten, _dctbasis_unflatten)


__all__ = ["DCTBasis"]
