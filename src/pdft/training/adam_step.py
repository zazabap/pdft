"""JIT'd fused Adam step for the batched training fast path.

This is intentionally a separate implementation from optimizers/adam.py:
it uses static lists indexed by k (XLA-friendly, no dict lookups inside
the JIT'd graph) instead of Python dicts keyed by manifold. The two
implementations must stay in trajectory-parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..loss import AbstractLoss, loss_function
from ..manifolds import UnitaryManifold, _make_identity_batch, group_by_manifold

Array = jax.Array


def _build_jit_adam_step(
    basis,
    loss: AbstractLoss,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    max_grad_norm: float | None,
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
    m_qb, n_qb = basis.m, basis.n
    code = basis.code
    inv_code = basis.inv_code

    def per_image_loss(tensors, img):
        return loss_function(tensors, m_qb, n_qb, code, img, loss, inverse_code=inv_code)

    batched_loss = jax.vmap(per_image_loss, in_axes=(None, 0))

    def stacked_loss(tensors, batch):
        return jnp.mean(batched_loss(tensors, batch))

    val_grad_fn = jax.value_and_grad(stacked_loss, argnums=0)

    # Pre-classify manifolds once — static across the whole training run.
    template_tensors = list(basis.tensors)
    groups = group_by_manifold(template_tensors)
    manifold_list = list(groups.keys())
    indices_list = [tuple(groups[mfd]) for mfd in manifold_list]

    # Pre-compute identity batches for unitary manifolds (closure constants
    # — avoids rebuilding them on every step inside `retract`).
    ibs = []
    for manifold, idxs in zip(manifold_list, indices_list):
        if isinstance(manifold, UnitaryManifold):
            d = template_tensors[idxs[0]].shape[0]
            ibs.append(_make_identity_batch(template_tensors[idxs[0]].dtype, d, len(idxs)))
        else:
            ibs.append(None)

    @jax.jit
    def step_fn(tensors_list, m_list, v_list, batch, lr, iter_1based):
        # Forward + backward; loss comes "for free" alongside grads.
        loss_val, raw_grads = val_grad_fn(tensors_list, batch)
        # Wirtinger conjugation: JAX returns ∂f/∂z̄, Julia Zygote returns ∂f/∂z.
        # See CLAUDE.md §1 — must stay or trajectories drift.
        grads = [jnp.conj(g) for g in raw_grads]

        # Per-manifold project (fused into the JIT'd graph).
        pb_list = []
        rg_list = []
        for manifold, idxs in zip(manifold_list, indices_list):
            pb = jnp.stack([tensors_list[i] for i in idxs], axis=-1)
            gb = jnp.stack([grads[i] for i in idxs], axis=-1)
            rg = manifold.project(pb, gb)
            pb_list.append(pb)
            rg_list.append(rg)

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
            rg = rg_list[k]
            pb = pb_list[k]
            m_old = m_list[k]
            v_old = v_list[k]

            new_m = beta1 * m_old + (1.0 - beta1) * rg
            new_v = beta2 * v_old + (1.0 - beta2) * jnp.real(jnp.conj(rg) * rg)

            direction = (new_m / bc1) / (jnp.sqrt(new_v / bc2) + eps)
            new_pb = manifold.retract(pb, -direction, lr, I_batch=ib)
            new_m = manifold.transport(pb, new_pb, new_m)

            new_m_list.append(new_m)
            new_v_list.append(new_v)
            for k2, idx in enumerate(idxs):
                # Use ellipsis so this works for any tensor storage shape:
                # (2, 2, n) for 2x2 unitaries, (2, 2, 2, 2, n) for 2-qubit
                # (Unitary2qManifold) gates, etc.
                new_tensors[idx] = new_pb[..., k2]

        return new_tensors, new_m_list, new_v_list, loss_val

    return step_fn
