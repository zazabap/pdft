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

from ._circuit import HADAMARD, Gate, compile_circuit, u4_from_phase

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


def _dense_qft_gates_1d(n_qubits: int, offset: int) -> list[Gate]:
    """Universal-for-SU(2^n) variant: QFT layout PLUS extra U(4) "wrap" gates
    that close the topology to a depth-equivalent of two QFT applications,
    enough to push the parameter count past dim(SU(2^n)).

    For n=3: 3 H + 6 U(4) = 9 + 90 = 99 free real params (vs 63 for SU(8)).
    This makes the parametric family contain DCT (and any other 8x8 unitary)
    EXACTLY — fit_to_dct can drive the residual to numerical zero, so
    DCT-init is meaningful.

    The extra U(4) gates use IDENTITY initialisation (trivially contained
    in U(4)) so the basis is still bit-identical to QFTBasis at training
    step 0.
    """
    import numpy as _np

    gates: list[Gate] = []
    # Layer 1: standard QFT topology (3 H + 3 U(4) at QFT phases)
    gates.extend(_rich_qft_gates_1d(n_qubits, offset=offset))
    # Layer 2: "wrap" U(4) gates on the same pairs, init to identity.
    eye_u4 = jnp.asarray(_np.eye(4).reshape(2, 2, 2, 2), dtype=jnp.complex128)
    for j in range(1, n_qubits + 1):
        for target in range(j + 1, n_qubits + 1):
            q_ctrl = offset + target
            q_tgt = offset + j
            gates.append(
                Gate(
                    kind="U4",
                    qubits=(q_ctrl, q_tgt),
                    tensor=eye_u4,
                    phase=0.0,
                )
            )
    return gates


def _rich_code(m: int, n: int, *, inverse: bool, dense: bool = False):
    """Build the rich (H + U(4)) circuit einsum + initial tensor list.

    ``dense=True`` uses the universal-coverage topology (twice as many U(4)
    gates per dim); ``dense=False`` uses the standard QFT layout.
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
    gates_fn = _dense_qft_gates_1d if dense else _rich_qft_gates_1d
    gates = gates_fn(m, offset=0) + gates_fn(n, offset=m)
    return compile_circuit(gates, m, n, inverse=inverse)


@dataclass
class RichBasis:
    """QFT topology with H + learnable U(4) gates instead of H + CP.

    Drop-in replacement for QFTBasis with strictly more expressivity:
    initialised identically (bit-equal forward output at training step 0)
    but the optimiser is free to deform the U(4) gates away from
    diagonal CP into any 4×4 unitary on each qubit pair.

    ``dense=True`` doubles the U(4) gate count (per-dim 3→6) by adding a
    second pass over the same qubit pairs initialised to identity. This
    pushes the parametric family past the SU(2^(m+n)) dimension threshold
    so the basis is provably universal — DCT (and any other unitary) is
    EXACTLY expressible. The basis is still bit-identical to QFTBasis at
    init because the extra gates are identities.

    Pytree contract:
        leaves   = tensors                                (one list)
        aux data = (m, n, len(tensors), code, inv_code, dense)
    """

    m: int
    n: int
    dense: bool
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
        dense: bool = False,
    ):
        if m < 1 or n < 1:
            raise ValueError(f"m and n must be >= 1, got m={m}, n={n}")
        self.m = m
        self.n = n
        self.dense = dense
        _code, init_tensors = _rich_code(m, n, inverse=False, dense=dense)
        _inv_code, _ = _rich_code(m, n, inverse=True, dense=dense)
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
    aux = (b.m, b.n, len(b.tensors), b.code, b.inv_code, b.dense)
    return leaves, aux


def _richbasis_unflatten(aux, leaves) -> RichBasis:
    m, n, n_fwd, code, inv_code, dense = aux
    assert len(leaves) == n_fwd
    return RichBasis(
        m=m, n=n, tensors=list(leaves), code=code, inv_code=inv_code, dense=dense
    )


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
    m: int,
    n: int,
    *,
    n_steps: int = 2000,
    lr: float = 0.02,
    dense: bool = False,
) -> list[Array]:
    """Fit RichBasis(m, n, dense=...) tensors such that the forward circuit
    ≈ DCT_2D.

    Target: ``DCT_{2^m} ⊗ DCT_{2^n}`` viewed as a (2^m, 2^n, 2^m, 2^n) tensor.
    Loss is the Frobenius norm between the circuit's action on a complete
    basis and that target. The output tensor list can be passed to
    ``RichBasis(m, n, tensors=..., dense=...)`` for a DCT warm-start.

    With ``dense=True`` the parametric family is universal for SU(2^(m+n)),
    so DCT IS reachable and the loss should drive to numerical zero. With
    ``dense=False`` (default, QFT topology) the family is too narrow at
    m=n=3 and the loss plateaus at ~63.7 — DCT is genuinely not in the
    family.
    """
    import time as _time

    from .manifolds import group_by_manifold, stack_tensors

    code, init_tensors = _rich_code(m, n, inverse=False, dense=dense)
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
