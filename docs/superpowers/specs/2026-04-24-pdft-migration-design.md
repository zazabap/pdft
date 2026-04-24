# pdft — Migration Design

**Date:** 2026-04-24
**Author:** zazabap (with Claude)
**Roadmap issue:** https://github.com/zazabap/pdft/issues/1
**Upstream:** https://github.com/nzy1997/ParametricDFT.jl

---

## 1. Purpose

Produce a **faithful Python reference** of ParametricDFT.jl so that:

1. Python (JAX/PyTorch/NumPy) users can consume and reproduce results from the Julia package.
2. Models and bases can be trained in one language and used in the other (bidirectional serialization).
3. Numerical behavior matches Julia within documented tolerances, verified continuously by committed golden vectors.

This is not a Pythonic rewrite. APIs and data layouts mirror Julia's wherever a one-to-one mapping exists.

## 2. Fixed decisions

| Decision | Choice | Rationale |
|---|---|---|
| Goal | Faithful reference / interop | Per user direction |
| Array + autodiff stack | **JAX** (`jax.grad`, `custom_vjp`, `vmap`, `jit`, `jnp.einsum`, `jnp.fft`, `jnp.linalg`) | Functional/immutable semantics mirror Julia; autodiff + GPU in one package; `custom_vjp` ≈ `ChainRulesCore.rrule`. Used **maximally** — no NumPy/SciPy at runtime. |
| Einsum path optimization | `jnp.einsum` + `jnp.einsum_path`, cached via `jax.jit` | XLA's contraction planner is effectively optimal for networks of this size; avoids an `opt_einsum` dep. If Phase 3 benchmarks show regression vs. Julia's `TreeSA`, we can reintroduce `opt_einsum` behind a feature flag. |
| Minimum Python | `>=3.11` | Tracks JAX 0.10.0's `requires_python` |
| CI matrix | 3.11 / 3.12 / 3.13 | Active security-supported Python lines, minus 3.14 (pending JAX wheel support) |
| Phasing | MVP vertical slice first (4 phases) | Ships working code early; each phase independently testable |
| Parity | Golden `.npz` vectors from a Julia harness in `reference/julia/` | Pins behavior to Julia; regenerated on demand, not in CI |
| Serialization | Bidirectional JSON compatible with Julia `StructTypes` schema | Enables cross-language round-trips |
| Circuit framework | **Hand-rolled** einsum skeletons | No Yao / Qiskit / PennyLane dep; circuits are fixed, known topologies |
| Package layout | Python files mirror `src/*.jl` filenames 1:1, created progressively per phase | Max traceability for Julia maintainers |
| Numerical precision | `complex128` / `float64` everywhere; `jax.config.update("jax_enable_x64", True)` set at `pdft` import time | Matches Julia's default `ComplexF64`; without x64, every parity test drifts by ~1e-7 and the tolerances in Section 6 are impossible to hit. The import-time side effect is documented in the README. |
| Upstream pinning | Exact `ParametricDFT.jl` git sha pinned in `reference/julia/Project.toml` and recorded in `manifest.json` on every golden regeneration | Prevents silent numerical drift when upstream moves |

## 3. Package layout

```
pdft/
├── src/pdft/
│   ├── __init__.py                  re-exports the public API per phase
│   ├── einsum_cache.py              jit-compiled einsum path cache (mirrors einsum_cache.jl)
│   ├── loss.py                      AbstractLoss, L1Norm, MSELoss, topk_truncate, loss_function
│   ├── qft.py                       qft_code, ft_mat, ift_mat                 ── Phase 1
│   ├── manifolds.py                 UnitaryManifold, PhaseManifold, batched ops, classify, stack/unstack  ── Phase 1
│   ├── optimizers.py                RiemannianGD (P1), RiemannianAdam (P2), optimize
│   ├── basis.py                     AbstractSparseBasis, QFTBasis (P1); Entangled/TEBD/MERA (P3)
│   ├── training.py                  train_basis, history recording, device placement  ── Phase 1
│   ├── io_json.py                   save_basis / load_basis, basis_to_dict / dict_to_basis, basis_hash  ── Phase 2
│   ├── entangled_qft.py  tebd.py  mera.py                              ── Phase 3
│   ├── compression.py                                                   ── Phase 4
│   └── viz.py  circuit_viz.py                                           ── Phase 4
├── tests/                           pytest; one test_<module>.py per src/<module>.py
├── reference/
│   ├── julia/                       tiny Julia harness; emits .npz + manifest.json
│   │   ├── Project.toml             pins ParametricDFT.jl at an exact sha + FFTW + JSON3 + NPZ.jl
│   │   ├── Manifest.toml            committed → reproducible resolve
│   │   ├── generate_goldens.jl      single entrypoint
│   │   └── cases/
│   │       ├── qft_code_4x4.jl
│   │       ├── qft_code_8x8.jl
│   │       ├── ft_mat_roundtrip.jl
│   │       ├── manifold_project_retract.jl
│   │       ├── loss_values.jl
│   │       └── train_trajectory_4x4.jl
│   └── goldens/                     committed outputs (<200 KB total)
└── examples/                        Pythonic demos (Phase 4)
```

**Filename mirror rule.** Each `src/<name>.jl` upstream has `src/pdft/<name>.py`, with exceptions:

| Julia | Python | Why |
|---|---|---|
| `ParametricDFT.jl` | `__init__.py` | Python package convention |
| `visualization.jl` | `viz.py` | Shorter |
| `circuit_visualization.jl` | `circuit_viz.py` | Shorter |
| `serialization.jl` | `io_json.py` | Avoid shadowing nothing in particular; explicit format |

Each Python file begins with `# Mirror of src/<name>.jl @ <upstream-sha>`.

**Naming conventions inside files.** snake_case functions (matching Julia), `PascalCase` classes (matching), constants in `UPPER_SNAKE`. Type hints use `jax.Array` with shape suffixes in docstrings (e.g., `U: Array[d, d, n]`).

## 4. Phase 1 module contracts (vertical slice)

**Phase 1 goal:** train a `QFTBasis` on a 4×4 image (2 qubits per axis) end-to-end with `RiemannianGD`, reproducing Julia's loss trajectory within `atol=1e-6`.

### `loss.py`

```python
class AbstractLoss(Protocol): ...

@dataclass(frozen=True)
class L1Norm(AbstractLoss): ...

@dataclass(frozen=True)
class MSELoss(AbstractLoss): ...

def topk_truncate(x: Array, k: int) -> Array:           # keep top-k magnitudes, zero rest
    ...

def loss_function(loss: AbstractLoss, pred: Array, target: Array) -> Array:   # scalar
    ...
```

### `einsum_cache.py`

```python
def optimize_code_cached(subscripts: str, *shapes) -> Callable[..., jax.Array]:
    """Return a jit-compiled einsum closure bound to fixed subscripts and shapes.

    Internally: compute `jnp.einsum_path(subscripts, *dummy_arrays_of_shape)` once,
    then return `jax.jit(lambda *ops: jnp.einsum(subscripts, *ops, optimize=path))`.
    Cached by (subscripts, shapes). Behaviorally replaces Julia's
    `optimize_code_cached(... TreeSA())`."""
```

### `qft.py`

```python
EinsumFn = Callable[..., jax.Array]   # jit-compiled closure returned by einsum_cache

def qft_code(m: int, n: int, *, inverse: bool = False) -> tuple[EinsumFn, list[Array]]:
    """Return (contraction_fn, initial_tensors). Tensors are ordered so all
    Hadamard-like gates come first (matches Julia's perm_vec sort).
    contraction_fn is a jit-compiled einsum bound to the fixed subscripts."""

def ft_mat (tensors, code, m, n, pic: Array[2**m, 2**n]) -> Array[2**m, 2**n]: ...
def ift_mat(tensors, code, m, n, pic: Array[2**m, 2**n]) -> Array[2**m, 2**n]: ...
```

### `manifolds.py`

```python
class AbstractRiemannianManifold(Protocol):
    def project  (self, points: Array, grads:   Array) -> Array: ...
    def retract  (self, points: Array, tangent: Array, alpha: float, *, I_batch=None) -> Array: ...
    def transport(self, old: Array, new: Array, vec: Array) -> Array: ...

@dataclass(frozen=True)
class UnitaryManifold(AbstractRiemannianManifold): ...    # U(n), Cayley retraction

@dataclass(frozen=True)
class PhaseManifold(AbstractRiemannianManifold): ...      # U(1)^d, normalize

def batched_matmul (A: Array, B: Array) -> Array          # jnp.einsum('ijk,jlk->ilk', ...)
def batched_adjoint(A: Array) -> Array                    # conj + transpose (0,1)
def batched_inv    (A: Array) -> Array                    # jnp.linalg.inv on (n,d,d) view
def stack_tensors  (tensors, idx: list[int]) -> Array[d, d, n]
def unstack_tensors(batch: Array, idx: list[int], into: list[Array]) -> None
def classify_manifold(t: Array) -> AbstractRiemannianManifold
def group_by_manifold(tensors: list[Array]) -> dict[AbstractRiemannianManifold, list[int]]
```

All batched operations use `(d, d, n)` layout, matching Julia exactly.

### `basis.py` (Phase 1 — `QFTBasis` only)

```python
class AbstractSparseBasis(Protocol):
    def forward_transform (self, pic: Array) -> Array: ...
    def inverse_transform (self, pic: Array) -> Array: ...
    @property
    def image_size(self) -> tuple[int, int]: ...
    @property
    def num_parameters(self) -> int: ...

@register_pytree_node_class
@dataclass
class QFTBasis(AbstractSparseBasis):
    m: int
    n: int
    tensors:     list[Array]                       # forward circuit
    inv_tensors: list[Array]                       # inverse circuit
    code:     EinsumFn = field(compare=False)  # jit-compiled einsum closure; static (not a pytree leaf)
    inv_code: EinsumFn = field(compare=False)
```

`QFTBasis` is a registered JAX pytree: its `tensors` / `inv_tensors` lists are leaves, and `(m, n, code, inv_code)` are aux data. This lets `jax.grad(loss_fn)(basis, ...)` traverse the basis transparently.

### `optimizers.py` (Phase 1 — `RiemannianGD` + `optimize` only)

```python
class AbstractRiemannianOptimizer(Protocol): ...

@dataclass
class RiemannianGD(AbstractRiemannianOptimizer):
    lr: float

def optimize(
    basis: AbstractSparseBasis,
    loss:  AbstractLoss,
    target: Array,
    opt:   AbstractRiemannianOptimizer,
    *, steps: int, seed: int,
) -> tuple[AbstractSparseBasis, list[float]]:
    """Runs `steps` iterations. Returns (updated_basis, loss_history)."""
```

### `training.py`

```python
@dataclass
class TrainingResult:
    basis:        AbstractSparseBasis
    loss_history: list[float]
    seed:         int
    steps:        int
    wall_time_s:  float

def train_basis(
    basis_ctor:  Callable[..., AbstractSparseBasis],
    target:      Array,
    loss:        AbstractLoss,
    optimizer:   AbstractRiemannianOptimizer,
    *, steps: int, seed: int = 0, device: str = "cpu",
) -> TrainingResult:
    ...
```

No plotting in Phase 1.

### Phase 1 public API (`__init__.py`)

```python
# Global: enable float64 / complex128 to match Julia's ComplexF64 default.
# Must precede any pdft import; documented as a side effect in the README.
import jax
jax.config.update("jax_enable_x64", True)

# Records the upstream ParametricDFT.jl commit this port is validated against.
# Updated atomically with golden regeneration (see reference/goldens/manifest.json).
__upstream_ref__ = "nzy1997/ParametricDFT.jl@<sha>"   # filled per-release
__version__      = "0.0.0"

from .loss       import AbstractLoss, L1Norm, MSELoss, topk_truncate, loss_function
from .qft        import qft_code, ft_mat, ift_mat
from .manifolds  import (AbstractRiemannianManifold, UnitaryManifold, PhaseManifold,
                         classify_manifold, group_by_manifold)
from .basis      import AbstractSparseBasis, QFTBasis
from .optimizers import AbstractRiemannianOptimizer, RiemannianGD, optimize
from .training   import train_basis, TrainingResult
```

## 5. Data flow (Phase 1 training loop)

```
┌───────────────────────────┐
│ target: Array[2^m, 2^n]   │   (e.g. FFTW-computed DFT of a 4×4 image)
│ basis:  QFTBasis          │   tensors, inv_tensors     ── pytree leaves
│                           │   code, inv_code           ── jit-compiled einsum closures (closure static)
│ opt:    RiemannianGD(lr)  │
└──────────────┬────────────┘
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1.  pred   = ft_mat(basis.tensors, basis.code, m, n, pic)              │
│  2.  scalar = loss_function(loss, pred, target)                         │
│  3.  euclidean_grads = jax.grad(step_loss)(basis.tensors, …)            │
│  4.  for each manifold M with indices idx:                              │
│         U  = stack_tensors(basis.tensors, idx)            (d,d,n)       │
│         G  = stack_tensors(euclidean_grads, idx)          (d,d,n)       │
│         Xi = M.project(U, G)                              (d,d,n)       │
│         U_new = M.retract(U, Xi, -lr, I_batch=cache)      (d,d,n)       │
│         unstack_tensors(U_new, idx, into=basis.tensors)                 │
│  5.  record scalar.item() in history                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**JAX-specific details.**

- The pure per-step function is `step(tensors, target) -> (new_tensors, loss)`. `basis` splits into `(tensors, static_rest)` so `jax.jit(step)` sees only array leaves.
- `code`, `inv_code`, `m`, `n`, `idx_U`, `idx_P` are closure constants (static), not leaves.
- `jax.grad` traverses the `tensors` list naturally; returns a same-shape list of Euclidean gradients. The jit-compiled `jnp.einsum` closure is fully differentiable under JAX.
- `I_batch` per manifold group is allocated once outside the step and closed over (matches Julia's pre-allocation strategy).
- GPU: `jax.device_put(tensors, jax.devices('gpu')[0])` before the loop. Step function is device-agnostic.

**Divergence from Julia.**

- Julia mutates `basis.tensors` in place; JAX is immutable — `step` returns a new list, outer loop rebinds. Structural shape of `QFTBasis` is invariant across steps.
- `batched_inv`: Julia uses a Python-level loop. JAX transposes `(d, d, n)` → `(n, d, d)`, calls `jnp.linalg.inv`, transposes back — truly batched.
- `classify_manifold` is called once at `QFTBasis.__post_init__`, not per step.

**Determinism.** `seed` creates `jax.random.PRNGKey(seed)`, threaded through any stochastic piece. `jnp.einsum_path` results are deterministic given fixed shapes; cached by `(subscripts, shapes)` in `einsum_cache.py`.

## 6. Parity harness

See Section 3 for layout. Summary of cases and tolerances:

| Case | Arrays saved | Tolerance |
|---|---|---|
| `qft_code_NxN` | `tensors_0..tensors_{T-1}`, `subscripts: str`, `iy: str`, `perm_vec: int[]` | shape-exact; tensors `atol=1e-12` |
| `ft_mat_roundtrip` | `pic`, `fwd`, `roundtrip` | `atol=1e-10` |
| `manifold_project_retract` | `U`, `G`, `Xi`, `U_new`, `alpha: float` | `atol=1e-10` |
| `loss_values` | `pred`, `target`, `l1`, `mse`, `topk_{k}` for k∈{1,3,5} | `atol=1e-12` |
| `train_trajectory_4x4` | `target`, `tensors_init_*`, `loss_history: float[51]`, `tensors_final_*`, config | loss history `atol=1e-6, rtol=1e-6`; final tensors `atol=1e-4` |

**Complex numbers.** `.npz` handles `complex128` natively; Julia's `NPZ.jl` writes the same format. No manual re-packing.

**Upstream commit pinning.** `reference/julia/Project.toml` records an exact `ParametricDFT.jl` git sha via `[compat]` + a Manifest.toml that pins the rev (e.g., `Pkg.add(url="...", rev="<sha>")`). The same sha is written to `reference/goldens/manifest.json` on every regeneration. A CI job (`.github/workflows/verify-upstream-pin.yml`) runs on PRs that touch `reference/` and fails if the pinned sha does not exist in the upstream repo history (one `gh api repos/nzy1997/ParametricDFT.jl/commits/<sha>` call). This catches accidentally-rebased or typo'd shas before they hit main.

**`manifest.json`.** Records SHA256 of each `.npz`, the pinned upstream commit sha, Julia version, timestamp, and (once we publish) the `pdft` version that produced this generation. Python tests *skip* (not fail) if the upstream sha in `reference/julia/Project.toml` diverges from the manifest — prevents silent drift after regeneration. The release process updates `pdft.__upstream_ref__` to match `manifest.json["upstream_sha"]` at tag time.

**Regeneration.** `make goldens` in repo root; needs Julia installed. Normal Python devs consume committed `.npz` files; they don't regenerate.

**Gauge caveat.** QFT tensors are unique only up to a global phase per gate. We pin to Julia's gauge by saving exact `tensors` from Julia's factorization. If this proves flaky, relax to "contracted circuit output matches within `atol=1e-10`" rather than element-wise tensor match.

**CI.** A single workflow runs Python 3.11/3.12/3.13 `pytest tests/`. A separate *manual-trigger* workflow regenerates goldens via a Julia action and opens a PR.

## 7. Test strategy

Three layers; every `src/pdft/<name>.py` has a matching `tests/test_<name>.py`.

**Layer 1 — Parity tests** (`tests/test_parity_*.py`): consume `.npz` goldens. Run in CI every push.

**Layer 2 — Property tests** (`tests/test_<module>.py`):

| Module | Properties |
|---|---|
| `qft.py` | `ft_mat ∘ ift_mat ≈ I` to `atol=1e-10`; equivalent to `jnp.fft.fft2` up to bit-reversal + normalization; tensor count matches gate count |
| `manifolds.py` | `U @ U.conj().T ≈ I` after `retract(U, project(U, G), α)` for random `G`, `α ∈ {1e-4, 1e-2, 1}`; projection output skew-Hermitian in `U'·Xi`; transport lives in tangent space at `U_new`. Phase manifold: retraction preserves unit modulus; projection is pure imaginary in `z̄·v` |
| `loss.py` | `L1Norm(x, x) == 0`; `MSELoss` non-negative, zero iff inputs equal; `topk_truncate(x, len(x))` is identity; `topk_truncate(x, 0)` is zero |
| `optimizers.py` | `RiemannianGD` decreases loss on random quadratic target for ≥80% of 100 steps; unitarity preserved across all 100 steps |
| `basis.py` | `QFTBasis` pytree round-trips (`flatten`/`unflatten`); `jax.grad` returns same-structure gradients |
| `training.py` | `loss_history[-1] ≤ loss_history[0]`; deterministic for fixed seed |

**Layer 3 — Smoke / integration** (`tests/test_smoke.py`, `tests/test_training_integration.py`).

**Budget.** Full suite <60s on a laptop (Phase 1). ≥90% line coverage on Phase-1 modules (`--cov-fail-under=90`).

**Randomness.** Explicit `jax.random.PRNGKey(<constant>)` per test. No implicit global state.

**Out of scope in Phase 1.** GPU numerics, plotting, serialization round-trip, bases beyond `QFTBasis`.

## 8. Error handling

This is a research library ported faithfully, not a user-facing service.

**Validate at public API boundaries; trust internal callers.**

- `QFTBasis(m, n)` — assert `m >= 1`, `n >= 1`.
- `ft_mat` / `ift_mat` — assert `pic.shape == (2**m, 2**n)`, complex dtype (auto-promote real).
- `loss_function(loss, pred, target)` — assert shapes match.
- `optimize(..., steps=k)` — assert `k >= 1`, `opt.lr > 0`.

Internal helpers (`batched_matmul`, `stack_tensors`, `classify_manifold`, einsum calls) trust their callers. No shape assertions in hot paths.

**Exception types.** `ValueError` for bad inputs; `TypeError` for wrong types; `AssertionError` reserved for internal invariants only, never at public surface.

**Numerical failures we handle.**

- Cayley retraction with singular `(I - α/2·W)` returns NaN silently. Training loop detects `jnp.any(jnp.isnan(loss))` at each step boundary and raises `RuntimeError("Training diverged at step {k}: loss is NaN. Try reducing lr.")`.
- Final `loss_history` must be finite; guard at `train_basis` return.

**Failures we don't defend against.**

- User passing already-NaN `pic` — final guard catches it.
- JAX platform mismatches — JAX's own errors are informative.

**Logging.** No `logging` calls in hot paths. `training.py` has an optional `verbose` + `log_every` knob that uses `print`. Silent by default.

**Error message style.** One line, include offending value + valid range: `ValueError("m must be >= 1, got m=0")`.

### 8.1 Conventions and invariants

**Complex dtype.** All public entry points accept real or complex `pic`. Internal representation is always `complex128` once `jax_enable_x64` is active (which `pdft` sets at import time). Real inputs are promoted to `complex128` at the API boundary; we never run a circuit in `complex64`. `tensors` and `inv_tensors` on `QFTBasis` are always `complex128`.

**Basis equality.** `QFTBasis` is a `dataclass` *and* a JAX pytree. Its `code` / `inv_code` fields are jit-compiled closures and compare by identity, so the dataclass-generated `__eq__` is useless for semantic comparison. `code` and `inv_code` are therefore declared `field(compare=False)`, and semantic equality is defined as: **same `m, n` and `jnp.allclose(tensors, tensors')` + `jnp.allclose(inv_tensors, inv_tensors')` element-wise under a stated tolerance.** A helper `pdft.basis.bases_allclose(a, b, *, atol=1e-10)` is provided for tests and serialization round-trip checks.

**Thread-safety.** `QFTBasis` is *not* thread-safe during training — `tensors` is rebound each step, and the update is not atomic. Training loops must not share a single `QFTBasis` across threads. Python 3.13 free-threading builds are supported for read-only use (inference via `ft_mat` / `ift_mat` on an already-trained basis), not for concurrent training.

## 9. Phased roadmap

### Phase 1 — QFT vertical slice
**Modules:** `einsum_cache.py`, `loss.py`, `qft.py`, `manifolds.py`, `basis.py` (QFTBasis), `optimizers.py` (RiemannianGD + `optimize`), `training.py`.
**Ports from:** `einsum_cache.jl`, `loss.jl`, `qft.jl`, `manifolds.jl`, slices of `basis.jl`, `optimizers.jl`, `training.jl`.
**Exit:** 4×4 training trajectory matches Julia golden (`atol=1e-6`); Layer-2 property tests green; CI 3.11/3.12/3.13 passing; ≥90% coverage on Phase-1 modules; `reference/julia/` harness working.

### Phase 2 — Full optimizer + serialization
**Modules added:** `RiemannianAdam` in `optimizers.py`; `io_json.py` (`save_basis`, `load_basis`, `basis_to_dict`, `dict_to_basis`, `basis_hash`).
**Ports from:** remainder of `optimizers.jl`, `serialization.jl`.
**Note on `PhaseManifold`:** its code ships in Phase 1 (as part of `manifolds.py`) and is covered there by property tests with synthetic data. `QFTBasis` produces only unitary gates, so the first *basis-integrated* use of `PhaseManifold` is Phase 3 (EntangledQFT / TEBD / MERA).
**Exit:** Adam parity golden matches Julia (loss `atol=1e-5`, final tensors `atol=1e-3`); Python-trained `QFTBasis` → serialize → load in Julia → identical `ft_mat` on a fixed image (and reverse); `basis_hash` matches byte-for-byte; JSON schema documented in `docs/serialization.md`.

### Phase 3 — Alternative circuit families
**Modules added:** `entangled_qft.py`, `tebd.py`, `mera.py`; `EntangledQFTBasis`, `TEBDBasis`, `MERABasis` in `basis.py`.
**Ports from:** `entangled_qft.jl`, `tebd.jl`, `mera.jl`, remaining `basis.jl`.
**Exit:** parity golden per basis (initial tensors + one training trajectory); `classify_manifold` returns expected manifold per gate type; unitarity preserved across GD and Adam for each family.

### Phase 4 — Compression + visualization + examples
**Modules added:** `compression.py`, `viz.py`, `circuit_viz.py`; `examples/` directory.
**Ports from:** `compression.jl`, `visualization.jl`, `circuit_visualization.jl`, `examples/*.jl`.
**Exit:** compression round-trip matches Julia pixel-wise (`atol=1e-8`); plotting functions produce PNGs on CI; ≥3 runnable examples (`basis_demo.py`, `mera_demo.py`, `optimizer_benchmark.py`), each <60s on CPU; `v0.1.0` published to PyPI.

**Dependencies between phases.** Phase 2 → Phase 1. Phase 3 → Phase 2. Phase 4 → Phase 3. Each phase has its own implementation plan via `superpowers:writing-plans`.

**Deferred (post-v0.1).** GPU-specific numerics tests and benchmarks (needs self-hosted runner). Python 3.14 support (blocked on JAX wheels). Docs site (MkDocs/Sphinx) — README-only until v0.1.

## 10. Open questions

None blocking Phase 1 start. The gauge caveat in Section 6 may require a relaxation once Julia tensors are committed; decide when the first golden is regenerated.
