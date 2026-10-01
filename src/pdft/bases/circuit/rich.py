"""RichBasis: single-layer parametric circuit with full 2-qubit unitary gates.

Motivation: at small block sizes (m=n=3 = 8x8) the existing H+CP gate
family hits an expressivity ceiling — all topologies converge ~1.75 dB
below 8x8 DCT. The cause is that diagonal CP gates have only 1 free
parameter each. A general 2-qubit unitary (U(4)) has 15 free parameters,
and the H + U(4) gate family is provably universal for SU(2^n) at any
qubit count >= 2, so it CONTAINS DCT as a special case.

RichBasis emits the same QFT topology gate sequence (H per qubit + 2-qubit
gates between qubit pairs) but each 2-qubit gate is a learnable U(4)
instead of a 1-parameter CP. Initialised so the circuit is BIT-IDENTICAL
to QFT at training step 0 (each U(4) gate equals the 4×4 controlled-phase
diag(1, 1, 1, exp(iφ)) at its standard QFT phase). This gives Adam a
gentle starting point: the optimiser begins exactly where plain QFT does
and can only improve.

Parameter count at m=n=3 (8x8 block):
  - 6 H gates (3 per dim) at 4 real params each = 24
  - 6 U(4) gates (3 per dim) at 15 real params each = 90
  - total: 114 real params per dim
  vs SU(8) dimension = 63 free real params
  → strictly more parameters than needed for any 8x8 unitary.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from ...circuit.builder import Gate, two_registers, u4_gate
from ...optimizers import RiemannianAdam, optimize
from ..core import CircuitBasis
from .qft import qft_gates_1d

Array = jax.Array


def _rich_qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """Same QFT topology as qft.qft_gates_1d, but with U(4) gates instead of CP.

    Each U(4) gate is initialised to the 4x4 unitary equivalent of the
    standard QFT phase (so the basis is bit-identical to QFTBasis at init).
    """
    return qft_gates_1d(n_qubits, offset, u4_gate)


def rich_gates(m: int, n: int) -> list[Gate]:
    """The gate sequence of the rich (H + U(4)) circuit on (2^m, 2^n) images."""
    return two_registers(_rich_qft_gates_1d, m, n)


class RichBasis(CircuitBasis):
    """QFT topology with H + learnable U(4) gates instead of H + CP.

    Parameter count at m=n=3 (within an 8x8 block):
      - 3 H gates per dim × 3 free real params (SU(2)) = 9
      - 3 U(4) gates per dim × 15 free real params (SU(4)) = 45
      - total per dim: 54 (BELOW the 63-dim of SU(8) — meaningful structure)

    Initialised so the circuit is BIT-IDENTICAL to QFTBasis at training step 0
    (each U(4) starts at the 4×4 controlled-phase diag(1, 1, 1, exp(iφ)) of
    its corresponding QFT slot). Adam can then deform the U(4) gates into any
    4×4 unitary, but the family is a strict 54-dim submanifold of SU(8) and
    does NOT contain 8×8 DCT exactly (empirically: fit_to_dct plateaus at
    Frobenius² ≈ 63.7).
    """

    emit = staticmethod(rich_gates)
    freezes_to_blocked = True


def _dct_matrix(n: int) -> Array:
    """Orthonormal 1D DCT-II matrix of size n × n."""
    import numpy as _np

    k = _np.arange(n).reshape(-1, 1)
    j = _np.arange(n)
    M = _np.cos(_np.pi * (2 * j + 1) * k / (2 * n))
    M[0, :] *= 1.0 / _np.sqrt(n)
    M[1:, :] *= _np.sqrt(2.0 / n)
    return jnp.asarray(M, dtype=jnp.complex128)


def fit_to_dct(
    basis_factory,
    *,
    n_steps: int = 2000,
    lr: float = 0.02,
) -> list[Array]:
    """Fit a parametric basis so its forward circuit ≈ DCT_2D.

    Parameters
    ----------
    basis_factory : callable
        Zero-argument callable returning a basis instance. Must expose
        ``.m, .n, .tensors, .code`` (any pdft basis class). The returned
        tensor list has the same shapes as ``basis_factory().tensors``
        and can be passed back as ``tensors=...`` for a DCT warm-start.
    n_steps : int
        Riemannian Adam steps (the package's own `optimize` loop). 2000 is a generous default; convergence is typically
        much faster when DCT lies in the parametric family.
    lr : float
        Adam learning rate.

    The loss is the Frobenius² distance between the circuit's action on
    a complete basis and ``DCT_{2^m} ⊗ DCT_{2^n}``. If the family does
    not contain DCT, the loss plateaus at the closest reachable distance.
    """
    basis = basis_factory()
    m, n = basis.m, basis.n
    code = basis.code
    D_row = _dct_matrix(2**m)
    D_col = _dct_matrix(2**n)
    target = jnp.kron(D_row, D_col).reshape(2**m, 2**n, 2**m, 2**n)

    eye = jnp.eye(2 ** (m + n), dtype=jnp.complex128).reshape((2 ** (m + n),) + (2,) * (m + n))
    target_flat = target.reshape(2**m, 2**n, 2 ** (m + n))

    def loss_fn(tensors):
        outs = jax.vmap(lambda x: code(*tensors, x))(eye)
        outs_mat = outs.reshape(2 ** (m + n), 2**m, 2**n).transpose(1, 2, 0)
        return jnp.real(jnp.sum(jnp.abs(outs_mat - target_flat) ** 2))

    value_and_grad = jax.value_and_grad(loss_fn)
    step = 0
    t0 = time.perf_counter()

    def grad_fn(tensors):
        # `optimize` asks for one gradient per step, so the loss that comes
        # with it is the progress report.
        nonlocal step
        step += 1
        loss_val, grads = value_and_grad(tensors)
        if step % 200 == 0 or step == 1:
            print(
                f"  fit_to_dct step {step:>4d}: loss={float(loss_val):.4e} "
                f"(elapsed {time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        return grads

    tensors, _ = optimize(
        RiemannianAdam(lr=lr), list(basis.tensors), loss_fn, grad_fn, max_iter=n_steps, tol=0.0
    )
    return tensors


__all__ = ["RichBasis", "fit_to_dct"]
