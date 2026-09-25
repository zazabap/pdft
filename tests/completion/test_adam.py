import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.completion.adam import adam_init, adam_update, apply_updates


def _reference(params, grads_seq, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Adam as written in Kingma & Ba, in numpy, one real leaf."""
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    p = params.copy()
    for t, g in enumerate(grads_seq, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        p = p - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    return p


def test_matches_the_textbook_update():
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal(7)
    grads = [rng.standard_normal(7) for _ in range(12)]
    params = jnp.asarray(p0)
    state = adam_init(params)
    for g in grads:
        upd, state = adam_update(jnp.asarray(g), state, 0.05)
        params = apply_updates(params, upd)
    assert int(state.count) == 12
    assert np.allclose(np.asarray(params), _reference(p0, grads, 0.05), atol=1e-13)


def test_complex_leaves_use_the_squared_modulus():
    rng = np.random.default_rng(1)
    params = {"a": jnp.asarray(rng.standard_normal(3) + 1j * rng.standard_normal(3))}
    g = {"a": jnp.asarray(rng.standard_normal(3) + 1j * rng.standard_normal(3))}
    state = adam_init(params)
    assert state.nu["a"].dtype == jnp.float64 and state.mu["a"].dtype == jnp.complex128
    upd, state = adam_update(g, state, 0.1)
    assert jnp.allclose(state.nu["a"], 0.001 * jnp.abs(g["a"]) ** 2)
    # first step of Adam moves every entry by lr in the direction of -g/|g|
    assert jnp.allclose(jnp.abs(upd["a"]), 0.1, atol=1e-6)
    new = apply_updates(params, upd)
    assert new["a"].dtype == params["a"].dtype


def test_agreement_with_optax():
    """Under jit, real leaves reproduce optax to the bit; the complex leaf may
    differ by an ulp on CPU (optax keeps a complex-typed second moment and
    takes a complex square root of it)."""
    optax = pytest.importorskip("optax")
    rng = np.random.default_rng(2)
    params = {
        "phi": jnp.asarray(rng.standard_normal((5, 4))),
        "blk": jnp.asarray(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))),
    }
    opt = optax.adam(3e-3)
    st_ref, st = opt.init(params), adam_init(params)
    p_ref, p = params, params

    @jax.jit
    def ours(p, st, g):
        u, st = adam_update(g, st, 3e-3)
        return apply_updates(p, u), st

    @jax.jit
    def theirs(p, st, g):
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    for _ in range(50):
        g = jax.tree.map(
            lambda x: (
                jnp.asarray(rng.standard_normal(x.shape)) * (1 + 0j if jnp.iscomplexobj(x) else 1)
            ),
            params,
        )
        p, st = ours(p, st, g)
        p_ref, st_ref = theirs(p_ref, st_ref, g)
    assert np.array_equal(np.asarray(p["phi"]), np.asarray(p_ref["phi"]))
    assert np.allclose(np.asarray(p["blk"]), np.asarray(p_ref["blk"]), rtol=0, atol=1e-14)
