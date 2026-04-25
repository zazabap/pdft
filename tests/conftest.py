"""Pytest session setup: enable JAX x64 before any pdft import.

Without this, JAX operates in float32/complex64 and parity tolerances are
impossible to hit. Importing pdft does this too (see src/pdft/__init__.py),
but we set it here as well so property tests that use `jax.numpy` directly
(without importing pdft) also run in x64.
"""

import jax

jax.config.update("jax_enable_x64", True)
