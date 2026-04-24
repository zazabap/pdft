import pdft


def test_version_string():
    assert isinstance(pdft.__version__, str)


def test_x64_enabled():
    import jax.numpy as jnp
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64
