"""Task-adapted training, and the compression control it must beat.

Compression training fits theta so that the basis reconstructs *fully observed*
images from their k largest coefficients; it never sees a mask. Task-adapted
training differentiates through the solver at the sampling rate we intend to
deploy at:

    theta* = argmin_theta  E_{(X,Omega)} || Xhat_K(theta; P_Omega X, Omega) - X ||_F^2 .

dim(theta) = n(n-1) ~ 72 at n = 9 against ~260k pixels, so the optimisation is
tiny and very over-determined.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .adam import adam_init, adam_update, apply_updates
from .solver import _THRESH, reconstruct_batch
from .transform import analysis, coherence, synthesis, theta0


def init_params(n: int) -> dict:
    """Start at the Fourier point: theta0 is an exact classical transform, so
    training deforms the FFT factorisation itself rather than appending a
    learned correction to it."""
    return {"r": theta0(n), "c": theta0(n)}


def task_loss(params, X, obs, n, k, K, mode, remat=True):
    """Loss of the K-step reconstruction from masked observations."""
    Xh = reconstruct_batch(params["r"], params["c"], X * obs, obs, n, k, K, mode, remat)
    return jnp.mean((Xh - X) ** 2)


def comp_loss(params, X, obs, n, k, K, mode, remat=True):
    """Reconstruct fully observed images from k coefficients. The mask is
    accepted and ignored, so the two objectives share a call signature."""
    del obs, K, remat
    C = _THRESH[mode](analysis(X, params["r"], params["c"], n), k)
    R = jnp.real(synthesis(C, params["r"], params["c"], n))
    return jnp.mean((R - X) ** 2)


_OBJECTIVES = {"task": task_loss, "comp": comp_loss}


def adam_loop(
    images,
    params,
    loss_fn,
    *,
    lr,
    steps,
    p,
    batch=2,
    seed=0,
    grad_mask=None,
    monitor=None,
    log_every=10,
    verbose=True,
):
    """The Adam loop every trained family shares; only the loss differs.

    Masks are redrawn every step at rate p, so the basis adapts to the sampling
    *rate*, not to one realisation of Omega. ``loss_fn(params, X, obs)`` is the
    objective; ``grad_mask``, if given, zeroes gradient entries so tied
    parameters stay at their initial value (Model A rides Model B's circuit
    this way); ``monitor(params)`` contributes extra fields (mu, typically) to
    the logged records. Raises on a non-finite gradient rather than training
    on garbage.

    Returns (params, history). Every family that trains by plain Adam goes
    through here so the batch/mask draw order --- and with it every seed ---
    is defined once. Manifold optimisers (Cayley SGD) cannot: their update is
    not an Adam transform.
    """
    opt_state = adam_init(params)

    @jax.jit
    def step(params, opt_state, X, obs):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, obs)
        if grad_mask is not None:
            grads = jax.tree.map(lambda g, m: g * m, grads, grad_mask)
        updates, opt_state = adam_update(grads, opt_state, lr)
        return apply_updates(params, updates), opt_state, loss, grads

    rng = np.random.default_rng(seed)
    history = []
    for it in range(steps):
        idx = rng.choice(len(images), size=min(batch, len(images)), replace=False)
        X = images[idx]
        obs = jnp.asarray(rng.random(X.shape) < p)
        params, opt_state, loss, grads = step(params, opt_state, X, obs)

        # jnp.abs, not g**2: some families carry complex leaves, for which g**2
        # is complex and its sum is not a norm.
        gnorm = float(jnp.sqrt(sum(jnp.sum(jnp.abs(g) ** 2) for g in jax.tree.leaves(grads))))
        if not np.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient at step {it}")
        rec = {"step": it, "loss": float(loss), "grad_norm": gnorm}
        if it % log_every == 0 or it == steps - 1:
            if monitor is not None:
                rec.update(monitor(params))
            if verbose:
                extra = "  ".join(
                    f"{a} {v:.4f}" for a, v in rec.items() if a not in ("step", "loss", "grad_norm")
                )
                print(f"  {it:4d}  loss {rec['loss']:.6e}  |g| {gnorm:.3e}  {extra}", flush=True)
        history.append(rec)

    return params, history


def train(
    images,
    n,
    k,
    K=20,
    p=0.10,
    steps=200,
    lr=2e-3,
    mode="hard",
    objective="task",
    batch=2,
    seed=0,
    lam_mu=0.0,
    log_every=10,
    remat=True,
    verbose=True,
):
    """The phase-only family trained through the solver (or the compression
    control). See :func:`adam_loop`."""
    loss_fn = _OBJECTIVES[objective]

    def total(params, X, obs):
        loss = loss_fn(params, X, obs, n, k, K, mode, remat)
        if lam_mu:
            # Nothing in the task objective prevents training from raising
            # mu, trading well-posedness for a lower loss on the sampling
            # pattern it saw. (For this family mu == 1 identically, so the
            # term is a control, not a need.)
            loss = loss + lam_mu * (coherence(params["r"], n) + coherence(params["c"], n))
        return loss

    def monitor(params):
        return {
            "mu_r": float(coherence(params["r"], n)),
            "mu_c": float(coherence(params["c"], n)),
        }

    return adam_loop(
        jnp.asarray(images),
        init_params(n),
        total,
        lr=lr,
        steps=steps,
        p=p,
        batch=batch,
        seed=seed,
        monitor=monitor,
        log_every=log_every,
        verbose=verbose,
    )
