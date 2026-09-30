"""Plain Adam on an unconstrained pytree, written out.

The completion trainers move circuit phases by ordinary Adam: the angles cover
the torus through ``exp(i*phi)``, so there is no manifold to retract onto, and
:class:`pdft.RiemannianAdam`, which moves tensors on their unitary manifolds,
is the wrong tool. The reference results were produced with ``optax.adam``;
this package does not depend on optax, so the update is written out here and
mirrors optax's arithmetic operation for operation: moment order, bias
correction computed in float64 and cast to the moment's dtype,
``sqrt(nu + 0) + eps``, the ``-lr`` scaling applied last, the update cast back
to the parameter's dtype. Under ``jax.jit``, which is how every trainer runs
it, a real-valued parameter tree reproduces optax to the bit on CPU and GPU;
a test checks that when optax happens to be installed. Eagerly, outside jit,
XLA fuses the two differently and they part by rounding (about 1e-15).

Complex leaves (the free-block butterfly) get a complex first moment and the
elementwise ``|g|^2`` as the second, as in optax. Optax carries that second
moment in the complex dtype and takes a complex square root of it, which on
CPU differs from the real square root here by an ulp; the test allows that.
Like optax, ``adam_update`` takes the gradient as given, so for a complex leaf
the caller passes the Euclidean gradient, ``conj(jax.grad(...))``, which
:func:`pdft.completion.training.adam_loop` does.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class AdamState(NamedTuple):
    count: Array
    mu: object
    nu: object


def _abs_sq(g: Array) -> Array:
    return (g.real**2 + g.imag**2) if jnp.iscomplexobj(g) else g**2


def adam_init(params) -> AdamState:
    """Zero moments of the parameter tree and an int32 count of 0, as optax."""
    return AdamState(
        count=jnp.zeros([], jnp.int32),
        mu=jax.tree.map(jnp.zeros_like, params),
        nu=jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.real(p).dtype), params),
    )


def adam_update(
    grads, state: AdamState, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
) -> tuple[object, AdamState]:
    """One Adam step: ``(updates, new_state)``, the updates to be added with ``apply_updates``."""
    mu = jax.tree.map(lambda g, m: (1 - b1) * g + b1 * m, grads, state.mu)
    nu = jax.tree.map(lambda g, v: (1 - b2) * _abs_sq(g) + b2 * v, grads, state.nu)
    count = state.count + 1
    bc1, bc2 = 1 - b1**count, 1 - b2**count
    mu_hat = jax.tree.map(lambda m: m / bc1.astype(m.dtype), mu)
    nu_hat = jax.tree.map(lambda v: v / bc2.astype(v.dtype), nu)
    updates = jax.tree.map(lambda m, v: -lr * (m / (jnp.sqrt(v + 0.0) + eps)), mu_hat, nu_hat)
    return updates, AdamState(count=count, mu=mu, nu=nu)


def apply_updates(params, updates):
    """``params + updates``, each leaf cast back to its parameter's dtype."""
    return jax.tree.map(
        lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params, updates
    )
