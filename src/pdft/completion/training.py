"""Task-adapted training: the pieces every trained family shares.

Compression training fits a basis so that it reconstructs fully observed
images from their ``k`` largest coefficients; it never sees a mask.
Task-adapted training differentiates through the solver at the sampling rate
the basis will be deployed at:

    theta* = argmin_theta  E_{(X, Omega)} || Xhat_K(theta; P_Omega X, Omega) - X ||_F^2 .

This module holds what that needs once: the minibatch and mask schedule (so
one seed means one trajectory for every family), the task loss of any solver,
the coherence monitor and the Adam loop. A family adds its parameters and its
operator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from .adam import adam_init, adam_update, apply_updates
from .transform import register_width

Array = jax.Array


def widths(images: Array) -> tuple[int, int]:
    """The ``(nr, nc)`` register widths of a stack of images."""
    return tuple(register_width(s) for s in images.shape[-2:])


def minibatches(
    images: Array, steps: int, batch: int, p: float, seed: int
) -> Iterator[tuple[Array, Array]]:
    """``steps`` draws of ``(X, obs)``: a batch without replacement and a fresh mask at rate ``p``.

    The mask is redrawn every step, so a basis adapts to the sampling rate
    rather than to one realisation of ``Omega``. Every trainer consumes this
    schedule, so a seed identifies the same batches for every family.
    """
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        X = images[rng.choice(len(images), size=min(batch, len(images)), replace=False)]
        yield X, jnp.asarray(rng.random(X.shape) < p)


def task_loss(
    reconstruct: Callable,
    params: dict,
    X: Array,
    obs: Array,
    k,
    K: int,
    mode: str = "hard",
    remat: bool = True,
) -> Array:
    """The mean squared error of a solver's ``K``-step recovery from the masked batch."""
    Xh = jax.vmap(lambda y, o: reconstruct(params["r"], params["c"], y, o, k, K, mode, remat))(
        X * obs, obs
    )
    return jnp.mean((Xh - X) ** 2)


def mu_monitor(coherence_fn: Callable) -> Callable:
    """A ``monitor`` for ``adam_loop`` that logs both axes' coherence."""
    return lambda params: {f"mu_{a}": float(coherence_fn(params[a])) for a in ("r", "c")}


def adam_loop(
    images: Array,
    params: dict,
    loss_fn: Callable,
    *,
    lr: float,
    steps: int,
    p: float,
    batch: int = 2,
    seed: int = 0,
    grad_mask: dict | None = None,
    monitor: Callable | None = None,
    log_every: int = 10,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Adam on ``params`` against ``loss_fn(params, X, obs)`` over ``minibatches``.

    ``grad_mask`` zeroes gradient entries so tied or frozen parameters stay at
    their initial value (Adam's update of a zero gradient is exactly zero).
    ``monitor(params)`` adds fields to the logged records. Raises on a
    non-finite gradient rather than training on garbage. Returns
    ``(params, history)``.

    The gradient is conjugated before it reaches Adam. For a complex leaf
    ``jax.grad`` returns the conjugate of the Euclidean gradient, and stepping
    along it unconjugated descends in the real parts while ascending in the
    imaginary ones; real leaves are untouched.
    """
    opt_state = adam_init(params)

    @jax.jit
    def step(params, opt_state, X, obs):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, obs)
        # jax.grad of a real loss in a complex input is the conjugate Wirtinger
        # derivative; conjugating recovers the Euclidean gradient.
        grads = jax.tree.map(jnp.conj, grads)
        if grad_mask is not None:
            grads = jax.tree.map(jnp.multiply, grads, grad_mask)
        updates, opt_state = adam_update(grads, opt_state, lr)
        return apply_updates(params, updates), opt_state, loss, grads

    history = []
    for it, (X, obs) in enumerate(minibatches(images, steps, batch, p, seed)):
        params, opt_state, loss, grads = step(params, opt_state, X, obs)
        # Squared moduli rather than squares: a complex leaf would make the sum complex.
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
