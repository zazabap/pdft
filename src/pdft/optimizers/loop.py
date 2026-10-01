"""The optimize() dispatcher: a single loop driving either GD or Adam."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp

from .adam import RiemannianAdam, _adam_step, _init_adam_state
from .core import _batched_project, _common_setup, _write_back
from .gd import RiemannianGD, _armijo_step

Array = jax.Array


def optimize(
    opt: RiemannianGD | RiemannianAdam,
    tensors: list[Array],
    loss_fn: Callable[[list[Array]], Array],
    grad_fn: Callable[[list[Array]], list[Array]],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    record_loss: bool = False,
    frozen_indices: frozenset[int] | None = None,
) -> tuple[list[Array], list[float]]:
    """Mirror of upstream src/optimizers.jl:335-412.

    Returns (final_tensors, loss_history). `loss_history` is empty unless
    `record_loss=True`; then it starts with the initial loss and appends
    one entry per iteration.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if opt.lr <= 0:
        raise ValueError(f"opt.lr must be > 0, got {opt.lr}")

    state = _common_setup(tensors)
    trace: list[float] = []
    if record_loss:
        trace.append(float(loss_fn(state.current_tensors)))

    cached_loss = float("nan")
    adam_state = _init_adam_state(state) if isinstance(opt, RiemannianAdam) else None

    for iter_0 in range(max_iter):
        _write_back(state)

        raw_grads = grad_fn(state.current_tensors)
        # JAX and Julia's Zygote use opposite Wirtinger conventions for gradients
        # of real-valued functions of complex inputs: JAX returns ∂f/∂z̄ while
        # Julia returns ∂f/∂z. These are complex conjugates. To match Julia's
        # trajectory (and produce correct updates w.r.t. the real manifold
        # structure), we conjugate the raw gradient before projection.
        raw_grads = [jnp.conj(g) for g in raw_grads]
        for g in raw_grads:
            if not bool(jnp.all(jnp.isfinite(g))):
                warnings.warn("Non-finite gradient — optimizer stopping.", stacklevel=2)
                return state.current_tensors, trace

        rg_batches, grad_norm = _batched_project(state, raw_grads, frozen_indices=frozen_indices)
        grad_norm_sq = float(grad_norm) ** 2

        if opt.max_grad_norm is not None and float(grad_norm) > opt.max_grad_norm:
            clip = opt.max_grad_norm / float(grad_norm)
            rg_batches = {m: b * clip for m, b in rg_batches.items()}
            grad_norm_sq = opt.max_grad_norm**2

        if float(grad_norm) < tol:
            break

        if isinstance(opt, RiemannianGD):
            cached_loss = _armijo_step(opt, state, rg_batches, loss_fn, grad_norm_sq, cached_loss)
        elif isinstance(opt, RiemannianAdam):
            _adam_step(opt, state, rg_batches, iter_0 + 1, adam_state)
        else:
            raise TypeError(f"unsupported optimizer type: {type(opt).__name__}")

        if record_loss:
            # The line search hands back the loss at the point it accepted.
            # Adam has none to hand back, and neither has a line search that ran
            # out and took its last candidate: read it at the new point.
            if jnp.isnan(cached_loss):
                _write_back(state)
                trace.append(float(loss_fn(state.current_tensors)))
            else:
                trace.append(float(cached_loss))

    _write_back(state)
    return state.current_tensors, trace
