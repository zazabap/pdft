"""The phase-only circuit: the completion paper's own family.

One Hadamard per wire and one controlled phase per wire pair, ``n(n-1)/2``
angles per axis, trained by plain Adam through the solver. The Hadamards are
fixed and every other gate is diagonal, so ``|U_ij| = N^{-1/2}`` for every
angle (Proposition 1 of the paper): coherence is pinned at 1 and nothing has to
maintain it. Training starts at ``theta0``, the DFT, so it deforms the FFT
factorisation rather than appending a learned correction to it.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..coherence import coherence, dense_operator
from ..solver import THRESHOLDS, batched, solver_for
from ..training import adam_loop, mu_monitor, task_loss, widths
from ..transform import apply_gates, n_from_params, separable, theta0, theta_to_params

Array = jax.Array


def apply_u(x: Array, theta: Array, adjoint: bool = False, axis: int = -1) -> Array:
    """Apply ``U(theta)``, or ``U^H``, along one axis."""
    return apply_gates(x, theta_to_params(theta), adjoint, axis)


analysis, synthesis = separable(apply_u)
reconstruct = solver_for(apply_u)
reconstruct_batch = batched(reconstruct)


def init_params(n: int) -> dict:
    """Both axes at the Fourier point."""
    return {"r": theta0(n), "c": theta0(n)}


def unitary_phases(theta: Array) -> Array:
    """``U(theta)`` formed explicitly. ``O(N^2 log N)``; diagnostics only."""
    return dense_operator(lambda e: apply_u(e, theta, axis=0), n_from_params(len(theta)))


def coherence_phases(theta: Array) -> Array:
    """``mu`` of ``U(theta)``: 1 for every ``theta``, and traceable, so a loss may carry it."""
    return coherence(unitary_phases(theta))


def comp_loss(
    params: dict, X: Array, obs: Array, k, K: int, mode: str = "hard", remat: bool = True
) -> Array:
    """The compression control: reconstruct fully observed images from ``k`` coefficients.

    The mask is accepted and ignored so the two objectives share a signature.
    """
    del obs, K, remat
    C = THRESHOLDS[mode](analysis(X, params["r"], params["c"]), k)
    return jnp.mean((jnp.real(synthesis(C, params["r"], params["c"])) - X) ** 2)


def train(
    images,
    k,
    K: int = 20,
    p: float = 0.10,
    steps: int = 200,
    lr: float = 2e-3,
    mode: str = "hard",
    objective: str = "task",
    batch: int = 2,
    seed: int = 0,
    lam_mu: float = 0.0,
    log_every: int = 10,
    remat: bool = True,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Train the angles through the solver (``objective="task"``) or by the compression control (``"comp"``).

    ``lam_mu`` adds a coherence penalty, a control this family never needs since
    its ``mu`` is pinned at 1. Returns ``(params, history)``.
    """
    images = jnp.asarray(images)
    nr, nc = widths(images)
    loss_fn = {"task": functools.partial(task_loss, reconstruct), "comp": comp_loss}[objective]

    def total(params, X, obs):
        loss = loss_fn(params, X, obs, k, K, mode, remat)
        if lam_mu:
            loss = loss + lam_mu * (coherence_phases(params["r"]) + coherence_phases(params["c"]))
        return loss

    return adam_loop(
        images,
        {"r": theta0(nr), "c": theta0(nc)},
        total,
        lr=lr,
        steps=steps,
        p=p,
        batch=batch,
        seed=seed,
        monitor=mu_monitor(coherence_phases),
        log_every=log_every,
        verbose=verbose,
    )
