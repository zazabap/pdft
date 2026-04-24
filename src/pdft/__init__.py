"""pdft: Python port of ParametricDFT.jl (faithful JAX reference).

Importing this package enables JAX's x64 mode globally. This is required
to match Julia's ComplexF64 numerical behavior; without it, parity
tolerances are unreachable. See docs/superpowers/specs/2026-04-24-pdft-migration-design.md
Section 2 and 8.1.
"""
import jax as _jax

_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.0"

# Filled at release time to the upstream commit sha that the committed
# goldens were generated against. See reference/goldens/manifest.json.
__upstream_ref__ = "nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef"
