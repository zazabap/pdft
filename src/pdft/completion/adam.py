"""Plain Adam on an unconstrained pytree, written out.

The completion trainers move the phases of a circuit by ordinary Adam: the
angles cover the torus through ``exp(i*phi)``, so there is no manifold to
retract onto and :class:`pdft.RiemannianAdam` (which updates *tensors* on
their unitary manifolds) is the wrong tool. The reference results were
produced with ``optax.adam``; this package does not take optax as a dependency,
so the update is written out here and mirrors optax's arithmetic operation for
operation --- moment order, bias correction computed in float64 then cast to
the moment's dtype, ``sqrt(nu + 0) + eps``, the ``-lr`` scaling applied last,
and the parameter update cast back to the parameter's dtype --- so a run here
reproduces a run there to the bit. A test compares the two when optax happens
to be installed.

Complex leaves (the free-block butterfly) are handled as optax handles them:
the first moment is complex, the second moment is the elementwise |g|^2.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class AdamState(NamedTuple):
    count: jax.Array
    mu: object
    nu: object


def _abs_sq(g):
    return (g.real**2 + g.imag**2) if jnp.iscomplexobj(g) else g**2


def adam_init(params) -> AdamState:
    """Zero moments of the parameter tree, count 0 (int32, as optax)."""
    return AdamState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree.map(jnp.zeros_like, params),
        nu=jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.real(p).dtype), params),
    )


def adam_update(
    grads, state: AdamState, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
):
    """One Adam step: returns ``(updates, new_state)``; add ``updates`` to the
    parameters with :func:`apply_updates`."""
    mu = jax.tree.map(lambda g, m: (1 - b1) * g + b1 * m, grads, state.mu)
    nu = jax.tree.map(lambda g, v: (1 - b2) * _abs_sq(g) + b2 * v, grads, state.nu)
    count = state.count + 1
    bc1 = 1 - b1**count
    bc2 = 1 - b2**count
    mu_hat = jax.tree.map(lambda m: m / bc1.astype(m.dtype), mu)
    nu_hat = jax.tree.map(lambda v: v / bc2.astype(v.dtype), nu)
    updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v + 0.0) + eps), mu_hat, nu_hat)
    updates = jax.tree.map(lambda u: -lr * u, updates)
    return updates, AdamState(count=count, mu=mu, nu=nu)


def apply_updates(params, updates):
    """``params + updates``, each leaf cast back to its parameter's dtype."""
    return jax.tree.map(
        lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params, updates
    )
