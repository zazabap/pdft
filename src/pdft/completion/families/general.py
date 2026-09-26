"""The relaxed QFT circuit of arXiv:2608.00053, and the ladder of models on it.

That work generalises the QFT by relaxing every gate within its own manifold:
"Each fixed Hadamard becomes an arbitrary unitary, and each conditional phase
keeps its diagonal form while freeing its diagonal entries." The kernel is
:func:`pdft.completion.transform.apply_gates`, which is also what
:class:`pdft.QFTBasis` trains under :func:`pdft.train_basis`
(:mod:`pdft.completion.bridge` converts between the two). Three nested models
ride it, all starting at the DFT:

    A  phases       fixed Hadamards, the controlled phase of each gate free
                    n(n-1)/2 params/axis, plain Adam, mu == 1 always
    B  diagonals    fixed Hadamards, all four phases of each gate free
                    2n(n-1) params/axis, plain Adam, mu == 1 always
    C  rotations    free U(2) per wire as well
                    4n + 2n(n-1) params/axis, Cayley step on U(2), mu drifts

A and B are ``train_general`` with different gradient masks; C is ``train_c``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..adam import adam_init, adam_update, apply_updates
from ..coherence import coherence, dense_operator
from ..solver import solver_for
from ..training import adam_loop, minibatches, mu_monitor, task_loss, widths
from ..transform import apply_gates, n_params, separable, theta0, theta_to_params
from .riemannian import cayley, skew

apply_general = apply_gates
analysis_g, synthesis_g = separable(apply_gates)
reconstruct_g = solver_for(apply_gates)


def init_general(n: int) -> dict:
    """The DFT: Hadamards, and phases (0, 0, 0, theta0)."""
    return theta_to_params(theta0(n))


def unitary_general(params: dict) -> jnp.ndarray:
    """The circuit's matrix, formed explicitly. Diagnostics only."""
    return dense_operator(lambda e: apply_gates(e, params, axis=0), params["g"].shape[0])


def coherence_general(params: dict) -> jnp.ndarray:
    return coherence(unitary_general(params))


def count_params(n: int) -> int:
    """Real DOF per axis of model C: U(2) per wire (4 each) + 4 phases per gate."""
    return 4 * n + 4 * n_params(n)


def count_params_b(n: int) -> int:
    """Real DOF per axis of model B. The one-qubit gates are not free, so they
    are not counted; freeing them is C, which loses Proposition 1."""
    return 4 * n_params(n)


def _mask(params: dict, model: str) -> dict:
    """Which entries move: B frees all four phases of every gate, A only the
    controlled one; the one-qubit gates stay at the Hadamards in both."""
    cols = slice(None) if model == "B" else 3
    return {
        "g": jnp.zeros_like(params["g"]),
        "phi": jnp.zeros_like(params["phi"]).at[:, cols].set(1.0),
    }


def train_general(
    images,
    k,
    K=100,
    p=0.10,
    steps=200,
    lr=2e-3,
    mode="hard",
    model="B",
    batch=2,
    seed=0,
    log_every=25,
    remat=True,
    verbose=True,
):
    """Models A and B: Adam on the phases through the solver, the Hadamards
    held by a gradient mask. Returns ``(params, history)``; ``params`` is the
    ``{"r", "c"}`` pair of ``{"g", "phi"}`` dicts the rest of the module reads."""
    if model not in ("A", "B"):
        raise ValueError(f"model must be 'A' or 'B', got {model!r}; model C is train_c")
    images = jnp.asarray(images)
    params = {a: init_general(n) for a, n in zip(("r", "c"), widths(images))}
    return adam_loop(
        images,
        params,
        functools.partial(task_loss, reconstruct_g, k=k, K=K, mode=mode, remat=remat),
        lr=lr,
        steps=steps,
        p=p,
        batch=batch,
        seed=seed,
        grad_mask={a: _mask(params[a], model) for a in params},
        monitor=mu_monitor(coherence_general),
        log_every=log_every,
        verbose=verbose,
    )


def train_c(
    images, k, K, p, steps, lr_phi, lr_g, seed, batch=2, log_every=50, verbose=True, remat=True
):
    """Model C: Adam on the phases, a Cayley step keeping each U(2) on its
    manifold. Same minibatch schedule as ``adam_loop``, so C's batches match
    A's and B's at the same seed. Returns ``params``.

    A check from exactly theta0 does not expose the sign of the Cayley step
    (the first-order term vanishes there and a top-k tie-break jump masks it);
    the tests verify descent from a perturbed point.
    """
    images = jnp.asarray(images)
    params = {a: init_general(n) for a, n in zip(("r", "c"), widths(images))}
    st = adam_init({a: params[a]["phi"] for a in params})
    loss = functools.partial(task_loss, reconstruct_g, k=k, K=K, remat=remat)

    @jax.jit
    def step(params, st, X, obs):
        v, grads = jax.value_and_grad(loss)(params, X, obs)
        upd, st = adam_update({a: grads[a]["phi"] for a in params}, st, lr_phi)
        phis = apply_updates({a: params[a]["phi"] for a in params}, upd)
        new = {
            a: {
                "g": cayley(params[a]["g"], skew(params[a]["g"], jnp.conj(grads[a]["g"])), lr_g),
                "phi": phis[a],
            }
            for a in params
        }
        return new, st, v

    for it, (X, obs) in enumerate(minibatches(images, steps, batch, p, seed)):
        params, st, v = step(params, st, X, obs)
        if verbose and (it % log_every == 0 or it == steps - 1):
            print(
                f"    {it:4d}  loss {float(v):.6e}  mu {float(coherence_general(params['r'])):.4f}",
                flush=True,
            )
    return params
