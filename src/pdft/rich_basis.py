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

from collections.abc import Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import tree_util

from .circuit.builder import HADAMARD, Gate, compile_circuit, u4_from_phase

Array = jax.Array


def _rich_qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """Same QFT topology as qft._qft_gates_1d, but with U(4) gates instead of CP.

    Each U(4) gate is initialised to the 4x4 unitary equivalent of the
    standard QFT phase (so the basis is bit-identical to QFTBasis at init).
    """
    gates: list[Gate] = []
    for j in range(1, n_qubits + 1):
        q = offset + j
        gates.append(Gate(kind="H", qubits=(q,), tensor=HADAMARD, phase=0.0))
        for target in range(j + 1, n_qubits + 1):
            k = target - j + 1
            t = offset + target
            phi = float(2 * jnp.pi / (2**k))
            gates.append(
                Gate(
                    kind="U4",
                    qubits=(t, q),  # control, target
                    tensor=u4_from_phase(phi),
                    phase=phi,
                )
            )
    return gates


def _rich_code(m: int, n: int, *, inverse: bool):
    """Build the rich (H + U(4)) circuit einsum + initial tensor list."""
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    gates = _rich_qft_gates_1d(m, offset=0) + _rich_qft_gates_1d(n, offset=m)
    return compile_circuit(gates, m, n, inverse=inverse)


@dataclass
class RichBasis:
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
        tensors: Sequence[Array] | None = None,
        code: object | None = None,
        inv_code: object | None = None,
    ):
        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        _code, init_tensors = _rich_code(m, n, inverse=False)
        _inv_code, _ = _rich_code(m, n, inverse=True)
        self.tensors = list(tensors) if tensors is not None else init_tensors
        self.code = code if code is not None else _code
        self.inv_code = inv_code if inv_code is not None else _inv_code

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


def _richbasis_flatten(b: RichBasis):
    leaves = tuple(b.tensors)
    aux = (b.m, b.n, len(b.tensors), b.code, b.inv_code)
    return leaves, aux


def _richbasis_unflatten(aux, leaves) -> RichBasis:
    m, n, n_fwd, code, inv_code = aux
    assert len(leaves) == n_fwd
    return RichBasis(m=m, n=n, tensors=list(leaves), code=code, inv_code=inv_code)


tree_util.register_pytree_node(RichBasis, _richbasis_flatten, _richbasis_unflatten)


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
        Adam steps. 2000 is a generous default; convergence is typically
        much faster when DCT lies in the parametric family.
    lr : float
        Adam learning rate.

    The loss is the Frobenius² distance between the circuit's action on
    a complete basis and ``DCT_{2^m} ⊗ DCT_{2^n}``. If the family does
    not contain DCT, the loss plateaus at the closest reachable distance.
    """
    import time as _time

    from .manifolds import group_by_manifold, stack_tensors

    basis = basis_factory()
    m, n = basis.m, basis.n
    code = basis.code
    init_tensors = basis.tensors
    D_row = _dct_matrix(2**m)
    D_col = _dct_matrix(2**n)
    target = jnp.kron(D_row, D_col).reshape(2**m, 2**n, 2**m, 2**n)

    eye = jnp.eye(2 ** (m + n), dtype=jnp.complex128).reshape((2 ** (m + n),) + (2,) * (m + n))
    target_flat = target.reshape(2**m, 2**n, 2 ** (m + n))

    def loss_fn(tensors):
        outs = jax.vmap(lambda x: code(*tensors, x))(eye)
        outs_mat = outs.reshape(2 ** (m + n), 2**m, 2**n).transpose(1, 2, 0)
        return jnp.real(jnp.sum(jnp.abs(outs_mat - target_flat) ** 2))

    grad_fn = jax.value_and_grad(loss_fn)

    tensors = list(init_tensors)
    groups = group_by_manifold(tensors)
    manifolds = list(groups.keys())
    indices = [tuple(groups[mfd]) for mfd in manifolds]

    # Adam state per group.
    m_state, v_state = [], []
    for idxs in indices:
        pb = stack_tensors(tensors, list(idxs))
        m_state.append(jnp.zeros_like(pb))
        v_state.append(jnp.zeros(pb.shape, dtype=jnp.float64))

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t0 = _time.perf_counter()
    for step in range(1, n_steps + 1):
        loss_val, raw_grads = grad_fn(tensors)
        grads = [jnp.conj(g) for g in raw_grads]

        for k, (mfd, idxs) in enumerate(zip(manifolds, indices)):
            pb = stack_tensors(tensors, list(idxs))
            gb = stack_tensors(grads, list(idxs))
            rg = mfd.project(pb, gb)
            new_m = beta1 * m_state[k] + (1 - beta1) * rg
            new_v = beta2 * v_state[k] + (1 - beta2) * jnp.real(jnp.conj(rg) * rg)
            bc1 = 1.0 - beta1**step
            bc2 = 1.0 - beta2**step
            direction = (new_m / bc1) / (jnp.sqrt(new_v / bc2) + eps)
            new_pb = mfd.retract(pb, -direction, lr)
            new_m = mfd.transport(pb, new_pb, new_m)
            m_state[k] = new_m
            v_state[k] = new_v
            for k2, idx in enumerate(idxs):
                tensors[idx] = new_pb[..., k2]

        if step % 200 == 0 or step == 1:
            print(
                f"  fit_to_dct step {step:>4d}: loss={float(loss_val):.4e} "
                f"(elapsed {_time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    return tensors


__all__ = ["RichBasis", "fit_to_dct"]
