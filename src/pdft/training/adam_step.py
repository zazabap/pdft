"""JIT'd fused Adam step for the batched training fast path.

A separate driver from `optimizers.optimize`, which takes one eager step at a
time with Python control flow on the gradient norm. This one fuses forward,
backward, projection, clipping, update and retraction into one compiled XLA
program with the learning rate and the step number traced.

The two drivers share the arithmetic: the update itself is
`optimizers.adam._adam_update`, the grouping is `optimizers.core._common_setup`
and the frozen-gradient stacking is `_stack_grads`. What differs is how the
arithmetic is run: op by op there, as one fused program here, with the clip
as a traced `minimum` instead of a Python branch. Their trajectories therefore
agree to rounding and not to the bit; `tests/training/test_batched.py` pins
how closely.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..loss import AbstractLoss, mean_loss
from ..manifolds import stack_tensors
from ..optimizers.adam import _adam_update, _zero_moments
from ..optimizers.core import _common_setup, _stack_grads

Array = jax.Array


def init_adam_moments(tensors) -> tuple[list[Array], list[Array]]:
    """Zero moment buffers for `tensors`: one ``(m, v)`` pair per manifold group.

    In the order the step built by `_build_jit_adam_step` for the same tensors
    takes them, since both group through `_common_setup`.
    """
    moments = [_zero_moments(pb) for pb in _common_setup(list(tensors)).point_batches.values()]
    return [m for m, _ in moments], [v for _, v in moments]


def _build_jit_adam_step(
    basis,
    loss: AbstractLoss,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    max_grad_norm: float | None,
    frozen_set: frozenset[int] | None = None,
):
    """Build a single JIT'd Adam step for `train_basis_batched`'s fast path.

    Returns ``step_fn(tensors_list, m_list, v_list, batch, lr_arr, iter_arr)
    -> (new_tensors_list, new_m_list, new_v_list, loss_value)``.

    The function fuses forward+backward+project+retract+transport+adam-update
    into one compiled XLA program. ``lr`` and ``iter_arr`` are TRACED inputs
    so the cosine schedule does not trigger XLA recompiles. All other Adam
    hyperparameters are compile-time constants.

    The Adam moment buffers (``m``, ``v``) are passed in/out so the caller
    persists them across steps — this is the corrected behaviour matching
    Julia's ``ParametricDFT.jl``: moments accumulate across the whole training
    run rather than being re-zeroed every batch.
    """
    val_grad_fn = jax.value_and_grad(mean_loss(basis, loss), argnums=0)

    # Group the tensors by manifold once: static across the whole training
    # run. The identity batches of the unitary manifolds become closure
    # constants, so `retract` does not rebuild them on every step.
    setup = _common_setup(list(basis.tensors))
    manifold_list = list(setup.manifold_groups)
    indices_list = [tuple(setup.manifold_groups[mfd]) for mfd in manifold_list]
    ibs = [setup.ibatch_cache.get(mfd) for mfd in manifold_list]
    # Static Python, closed over by the jitted step: no recompiles.
    frozen = frozenset(frozen_set or ())

    @jax.jit
    def step_fn(tensors_list, m_list, v_list, batch, lr, iter_1based):
        # Forward + backward; loss comes "for free" alongside grads.
        loss_val, raw_grads = val_grad_fn(tensors_list, batch)
        # Wirtinger conjugation: JAX returns ∂f/∂z̄, Julia Zygote returns ∂f/∂z.
        # See CLAUDE.md §1 — must stay or trajectories drift.
        grads = [jnp.conj(g) for g in raw_grads]

        # Per-manifold project (fused into the JIT'd graph).
        pb_list = [stack_tensors(tensors_list, idxs) for idxs in indices_list]
        rg_list = [
            manifold.project(pb, _stack_grads(grads, idxs, frozen))
            for manifold, idxs, pb in zip(manifold_list, indices_list, pb_list)
        ]

        # Optional global gradient clipping (compile-time branch).
        if max_grad_norm is not None:
            grad_norm_sq = jnp.zeros((), dtype=jnp.float64)
            for rg in rg_list:
                grad_norm_sq = grad_norm_sq + jnp.real(jnp.sum(jnp.conj(rg) * rg))
            grad_norm = jnp.sqrt(grad_norm_sq)
            clip = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-30))
            rg_list = [rg * clip for rg in rg_list]

        # Bias correction with TRACED iter — no recompile when iter changes.
        bc1 = 1.0 - beta1**iter_1based
        bc2 = 1.0 - beta2**iter_1based

        new_tensors = list(tensors_list)
        new_m_list = []
        new_v_list = []
        for k, (manifold, idxs, ib) in enumerate(zip(manifold_list, indices_list, ibs)):
            new_pb, new_m, new_v = _adam_update(
                manifold,
                pb_list[k],
                rg_list[k],
                m_list[k],
                v_list[k],
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                bc1=bc1,
                bc2=bc2,
                I_batch=ib,
            )
            new_m_list.append(new_m)
            new_v_list.append(new_v)
            # The last axis is the stack axis, whatever the tensor's rank. A
            # frozen tensor is handed back as it came in, bit for bit.
            for k2, idx in enumerate(idxs):
                if idx not in frozen:
                    new_tensors[idx] = new_pb[..., k2]

        return new_tensors, new_m_list, new_v_list, loss_val

    return step_fn
