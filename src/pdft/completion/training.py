"""Task-adapted training, and the compression control it must beat.

Compression training fits a basis so that it reconstructs *fully observed*
images from their k largest coefficients; it never sees a mask. Task-adapted
training differentiates through the solver at the sampling rate we intend to
deploy at:

    theta* = argmin_theta  E_{(X,Omega)} || Xhat_K(theta; P_Omega X, Omega) - X ||_F^2 .

Everything a trained family needs is here once: the minibatch and mask
schedule (so one seed means one trajectory for every family), the task loss
of any solver, the mu monitor, and the Adam loop.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .adam import adam_init, adam_update, apply_updates
from .solver import _THRESH, reconstruct
from .transform import analysis, coherence, register_width, synthesis, theta0


def init_params(n: int) -> dict:
    """Both axes at the Fourier point, so training deforms the FFT factorisation
    itself rather than appending a learned correction to it."""
    return {"r": theta0(n), "c": theta0(n)}


def widths(images) -> tuple[int, int]:
    """``(nr, nc)`` of a stack of images."""
    return tuple(register_width(s) for s in images.shape[-2:])


def minibatches(images, steps: int, batch: int, p: float, seed: int):
    """``steps`` draws of ``(X, obs)``: a batch of images without replacement
    and a fresh mask at rate ``p``, so the basis adapts to the sampling *rate*
    rather than to one realisation of Omega. The one schedule every trainer
    consumes."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        X = images[rng.choice(len(images), size=min(batch, len(images)), replace=False)]
        yield X, jnp.asarray(rng.random(X.shape) < p)


def task_loss(reconstruct, params, X, obs, k, K, mode="hard", remat=True):
    """Mean squared error of ``reconstruct``'s K-step recovery from the masked batch."""
    Xh = jax.vmap(lambda y, o: reconstruct(params["r"], params["c"], y, o, k, K, mode, remat))(
        X * obs, obs
    )
    return jnp.mean((Xh - X) ** 2)


def comp_loss(params, X, obs, k, K, mode="hard", remat=True):
    """The compression control for the phase-only family: reconstruct fully
    observed images from k coefficients. The mask is accepted and ignored, so
    the two objectives share a signature."""
    del obs, K, remat
    C = _THRESH[mode](analysis(X, params["r"], params["c"]), k)
    return jnp.mean((jnp.real(synthesis(C, params["r"], params["c"])) - X) ** 2)


def mu_monitor(coherence_fn):
    """A ``monitor`` for ``adam_loop`` that logs both axes' coherence."""
    return lambda params: {f"mu_{a}": float(coherence_fn(params[a])) for a in ("r", "c")}


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
    """Adam on ``params`` against ``loss_fn(params, X, obs)`` over ``minibatches``.

    ``grad_mask`` zeroes gradient entries so tied or frozen parameters stay at
    their initial value (Adam's update of a zero gradient is exactly zero);
    ``monitor(params)`` adds fields to the logged records. Raises on a
    non-finite gradient rather than training on garbage. Returns
    ``(params, history)``.
    """
    opt_state = adam_init(params)

    @jax.jit
    def step(params, opt_state, X, obs):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, obs)
        if grad_mask is not None:
            grads = jax.tree.map(jnp.multiply, grads, grad_mask)
        updates, opt_state = adam_update(grads, opt_state, lr)
        return apply_updates(params, updates), opt_state, loss, grads

    history = []
    for it, (X, obs) in enumerate(minibatches(images, steps, batch, p, seed)):
        params, opt_state, loss, grads = step(params, opt_state, X, obs)
        # |g|^2, not g**2: complex leaves would make the sum complex.
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
    """The phase-only family trained through the solver (``objective="task"``)
    or by the compression control (``"comp"``). ``lam_mu`` adds a coherence
    penalty, a control that this family, with mu pinned at 1, never needs."""
    images = jnp.asarray(images)
    nr, nc = widths(images)
    loss_fn = {"task": functools.partial(task_loss, reconstruct), "comp": comp_loss}[objective]

    def total(params, X, obs):
        loss = loss_fn(params, X, obs, k, K, mode, remat)
        if lam_mu:
            loss = loss + lam_mu * (coherence(params["r"]) + coherence(params["c"]))
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
        monitor=mu_monitor(coherence),
        log_every=log_every,
        verbose=verbose,
    )
