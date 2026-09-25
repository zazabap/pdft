# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Read this before making non-trivial changes — several conventions are *not* discoverable from the code without painful debugging, and at least one is documented here only because the original Python port spent hours rediscovering it.

## What this is

`pdft` is a faithful Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl) using JAX. Goal: a Python user can reproduce Julia results bit-for-bit (or within documented tolerances) on the same input. **Behavior parity with Julia is the primary correctness criterion** — never sacrifice it for Pythonic-ness.

- Design spec: `docs/superpowers/specs/2026-04-24-pdft-migration-design.md`
- Roadmap: GitHub issue #1
- Reliability hardening backlog: GitHub issue #2
- Upstream is pinned to `nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef`. The pin lives in two places that must stay in sync: `reference/julia/generate_goldens.jl` (`UPSTREAM_SHA`) and `src/pdft/__init__.py` (`__upstream_ref__`).

## Critical conventions (do not break)

These are the non-obvious invariants that took the original port multiple iterations to find. Violating any of them silently breaks parity with Julia.

### 1. Wirtinger gradient conjugation

**JAX returns `∂f/∂z̄` while Julia's Zygote returns `∂f/∂z`** for real-valued functions of complex inputs. They are complex conjugates. `optimize()` in `src/pdft/optimizers.py` conjugates `raw_grads` immediately after `grad_fn(...)`:

```python
raw_grads = grad_fn(state.current_tensors)
raw_grads = [jnp.conj(g) for g in raw_grads]   # MUST stay
```

Without this line, GD trajectories drift ~10% over 50 steps and Adam outright diverges. Any new optimizer added to this module must apply the same conjugation, or define a custom JAX `vjp` that matches Julia's convention.

### 2. Yao little-endian qubit ordering

`pic.reshape((2,)*(m+n))` gives axes in big-endian bit order, but Julia's Yao treats qubit 1 as the LSB. `_circuit.build_circuit_einsum` reverses qubit-to-axis mapping within each block:

```python
row_pic = [input_labels[q - 1] for q in range(m, 0, -1)]   # axis 0 ↔ qubit m
col_pic = [input_labels[q - 1] for q in range(m + n, m, -1)]
```

Same reversal applies to `out_labels`. Don't undo this without first making `ft_mat` parity tests fail in a controlled way.

### 3. Compact 2×2 CP tensor representation

A controlled-phase gate is `diag(1, 1, 1, exp(iφ))` as a 4×4 matrix, but Yao's `yao2einsum` emits the **compact 2×2 form**:

```python
controlled_phase_diag(φ) = [[1, 1], [1, exp(iφ)]]
```

This shares the wire labels of control and target qubits and does NOT introduce new labels (the gate is diagonal so each wire's value is unchanged; only a multiplicative phase is injected). All circuit modules use this representation; never substitute the 4×4 form even though it would be mathematically equivalent — Julia goldens won't match.

### 4. Hadamard-first tensor sort

Julia's `qft_code` does `perm_vec = sortperm(tn.tensors, by=x -> !(x ≈ mat(H)))` — Hadamards come first, CPs after. Python's `_circuit.build_circuit_einsum` applies the same sort. The einsum is rebuilt with the permuted tensor list AND permuted subscripts so it stays valid. **JSON serialization assumes this order**; cross-language interop breaks if you change it.

### 5. JAX x64 mode at import time

`src/pdft/__init__.py` calls `jax.config.update("jax_enable_x64", True)` before any other import. `tests/conftest.py` does the same. Without x64, JAX uses complex64 / float32 and parity tolerances become unreachable. Any new test file that uses `jax.numpy` directly without importing `pdft` first must include the same call (or import pdft to force it).

### 6. Column-major iteration for cross-language hashes

`io_json.basis_hash` and `compression` use **Julia 1-based column-major** flat indices. NumPy is row-major by default. Use `array.flatten(order="F")` or `np.unravel_index(..., order="F")`. Never use plain `array.flatten()` for serialization.

### 7. Julia-compatible float formatting

Python's `repr(5e-7)` gives `"5e-07"`; Julia's `string(5e-7)` gives `"5.0e-7"`. `io_json._format_float_julia_like` post-processes `repr()` to match Julia. Use it (not `repr` or `str`) for any value that participates in a cross-language hash or JSON byte-comparison.

### 8. L1-cusp horizon for trajectory parity

GD on L1 loss is bit-exact for ~50 steps and stays within `atol=1e-3` over 200 steps. Beyond that, FP accumulation eventually pushes Python and Julia across the same loss cusp at slightly different alphas in Armijo line search. Both still converge to the same loss basin. Don't tighten `tests/test_parity_long_run.py` to `atol=1e-10` over 200 steps — this is mathematically expected on non-smooth losses, not a bug. Smooth losses (MSE without truncation) don't have this problem.

### 9. Phase extractors classify by tensor shape, not gate-list metadata

`get_*_gate_indices` walks the tensor list and tags any 2×2 tensor whose four entries have unit-modulus magnitudes (within `atol=0.15`) as a CP gate, then returns the LAST `n_gates` such positions. After training, individual entries can drift slightly off the unit circle; the moderate tolerance accommodates that. If you tighten the tolerance, do so in tandem with a regression test on a trained basis.

## `pdft.completion` (ported from pdft-completion)

The library half of [pdft-completion](https://github.com/zazabap/pdft-completion),
the code behind *Image Inpainting from Random Pixels with a Trainable Quantum
Fourier Transform*, lives in `src/pdft/completion/`. It is **not a Julia port**:
there are no goldens for it, and its correctness criteria are the property
tests in `tests/completion/` (the paper repo's `verify.py` / `verify_unroll.py`
checks, turned into pytest) plus agreement with the paper repo's numbers. The
paper's experiment scripts, data fetchers, figures and `results/` stay in that
repo; only code two scripts would share was brought over, and every function
that took a repository path now takes an explicit directory.

It carries the QFT circuit in a **second representation**: the gate angles
(`transform.apply_u`) or the `{"g", "phi"}` gate dict (`families.general`),
applied directly to the image with no matrix formed, trained by plain Adam.
The conventions below are load-bearing; several were rediscovered painfully
upstream.

### 10. Sign convention: `U(theta0) == conj(DFT_ortho)`

The QFT uses `e^{+2 pi i kx/N}`, so the circuit at the textbook angles is the
*conjugate* of NumPy's orthonormal DFT. `tests/completion/test_transform.py`
pins it. Do not "fix" it: `synthesis` is `U`, `analysis` is `U^H`, and every
family (`butterfly.dft_blocks`, `riemannian.dft_matrix`) is anchored to the
same sign so the comparisons stay nested.

### 11. The bridge identity (QFTBasis <-> angles)

`QFTBasis` applies its gates in Yao's order (qubit 1 is the LSB and gets its
Hadamard first, no final swap layer). In the same axis frame that is the
**reverse** of the completion circuit's gate sequence, so the operators are
transposes of each other up to the bit reversal the completion circuit ends
with. Exactly, with `Pi` the bit-reversal permutation of an axis:

```
QFTBasis.forward_transform(X) == U(theta_r)^T (Pi X Pi) U(theta_c)
```

`completion/bridge.py` implements it: one-qubit gates map to their transposes
(a Hadamard is symmetric, so phase-only bases need none), the four phases of
each two-qubit gate are re-indexed to the other bit order
(`exp(i phi).reshape(2, 2).T`), and the gate index map comes from
`sorted_gate_program`, never from arithmetic on the emission order.
`tests/completion/test_bridge.py` checks the identity at random *non-symmetric*
gates; the phase-only case cannot catch a wrong transpose. `mu` needs no
adjustment (invariant under permutation, transposition and conjugation), so
`pdft.coherence.coherence` of a converted basis equals the product of the
per-axis `coherence_general` values, and that is tested too. Consequence
worth knowing: at initialisation `QFTBasis` is the DFT of the *bit-reversed*
image, not of the image (a fixed pixel permutation; inherited from Yao, and
what the Julia goldens encode).

### 12. Straight-through thresholds

`solver.hard_k` is `C * M(C)` with the support mask `M` under
`stop_gradient`; `soft_k` holds its threshold `lambda` the same way. `M` is
piecewise constant, so differentiating it naively kills the selection pathway
and the task loss goes flat. A silent zero gradient is the most likely failure
of the training objective; `test_solver.py` asserts the gradient is live.

### 13. No optax: `completion/adam.py` mirrors `optax.adam` operation for operation

Same rule as the rest of the package (no optax dependency), but here the
reference results *were* produced with `optax.adam`, so the written-out update
must reproduce it to the bit: moment order, bias correction in float64 then
cast to the moment's dtype, `sqrt(nu + 0) + eps`, `-lr` applied last, the
update cast back to the parameter dtype, `|g|^2` for complex leaves.
`tests/completion/test_adam.py` compares against optax when it happens to be
importable (it is not a dev dependency) and against the textbook formula
always. Every Adam-trained family goes through `training.adam_loop`, so the
batch/mask draw order, and with it every seed, is defined once. Cayley SGD
(`families.riemannian`, `general.train_c`) keeps its own loop by necessity.

### 14. The Cayley step descends with `+tau`

For a U(2) or U(N) gate `G` with Euclidean gradient `E = conj(jax.grad(...))`
(the Wirtinger conjugation again, see §1), the generator is
`A = G^H E - E^H G` and the descent step is `cayley(G, A, +tau)`. The sign was
wrong for a month upstream and nothing caught it, because a check from exactly
`theta0` has a vanishing first-order term and a top-k tie-break jump masks the
rest. `test_general.py::test_cayley_step_descends_from_a_perturbed_point`
starts from a perturbed point on purpose; keep it that way.

### 15. `k` is traced in the evaluation solvers

`solver.kth_largest` sends a Python-int `k` through `top_k` (training, `k`
fixed) and a traced `k` through a full sort (evaluation, `k = budget_k`
differs with every mask). Keep `k` traced in `reconstruct*`: recompiling the
K-step scan once per (image, budget) was the whole cost of scoring a sweep.
The two thresholds are identical.

### 16. Never form `U` on a hot path

`dense_operator`, `unitary_matrix`, `unitary_general`, `unitary_butterfly` cost
`O(N^2 log N)` and are diagnostics. Transforms apply gates to the image.

### 17. The image's dtype sets the working precision

`transform.complex_dtype` picks complex64 for float32 images; the angles stay
float64. The circuit has no matmul, so float32 loses nothing measurable, but
the families that do multiply matrices (transform learning, the butterfly's
blocks, `riemannian`) lose up to 0.55 dB under XLA's default TF32, which is
why `protocol.evaluate` and `table1_scores` run under
`jax.default_matmul_precision("highest")`, and why every matrix-valued solver
casts its matrices to the carry's dtype (a mismatched `lax.scan` carry is a
hard error). Tests that compare float32 contractions on a GPU need the same
context.

### 18. `metrics.gaussian_filter` must match scipy's

SSIM and MS-SSIM use an 11-tap Gaussian (`sigma = 1.5`, `truncate = 3.5`,
`mode = "reflect"`) written in numpy so scipy is not a dependency;
`test_metrics.py` pins it against `scipy.ndimage.gaussian_filter` to 1e-12
when scipy is installed. The paper's MS-SSIM column depends on it.

### Naming

`pdft.completion.coherence` the *attribute* is the theta-based function
re-exported from `transform` (the core package does the same with
`pdft.coherence`); the module is reached with `from pdft.completion.coherence
import ...` or `importlib.import_module`. The training module is `training`,
not `train`, so the `train` function does not shadow it.

## Repo layout

```
src/pdft/
├── manifolds.py            Mathematical core (UnitaryManifold, PhaseManifold, batched ops)
├── loss.py                 L1Norm, MSELoss, topk_truncate, loss_function (public)
├── profiling.py            Cross-cutting profiling helpers
├── bases/
│   ├── base.py             AbstractSparseBasis + bases_allclose + 4 basis dataclasses
│   ├── circuit/            QFT, EntangledQFT, TEBD, MERA, Rich, RealRich + freeze_as_blocked
│   └── block/              BlockedBasis (Rich/RealRich re-exported from circuit for back-compat)
├── circuit/                Einsum builder (builder.py) + JIT closure cache (cache.py)
├── optimizers/             core, gd (RiemannianGD + Armijo), adam (RiemannianAdam), loop
├── training/               schedules, single (train_basis), batched, adam_step, eval_loop
├── io/                     serialize (JSON), compression
├── viz/                    loss (matplotlib loss plots), circuit (schematic)
└── completion/             Image inpainting from random pixels (ported from pdft-completion; see below)
    ├── transform, solver, unroll, training, adam, coherence, metrics, protocol, data, bridge
    ├── families/           general (QFT + diagonals / rotations), shared, butterfly, riemannian
    └── baselines/          fixed_bases, nuclear, qtt, transform_learning

reference/julia/            Julia harness — needed only to regenerate goldens
reference/goldens/          Committed .npz + .json files (<200 KB total)
examples/                   4 runnable demos, each <10s
tests/                      pytest; mirrors src/pdft/ layout (tests/bases/, tests/optimizers/, ...)
```

Benchmarks live in a separate repo: https://github.com/zazabap/pdft-benchmarks (split from this repo at pdft v0.2.0; pinned via its `pyproject.toml`).

`DCTBasis` was removed during the modular-src refactor (PR #11). Use a parametric basis (`RichBasis` / `RealRichBasis`) for similar use cases.

`pdft/__init__.py` re-exports a small set of most-used names (basis classes,
trainer, optimizer, loss). Less-public names live in their subpackages —
e.g. `pdft.io.save_basis`, `pdft.manifolds.UnitaryManifold`,
`pdft.profiling.profile_training`, `pdft.bases.circuit.qft_code`.

`AbstractRiemannianOptimizer` is `RiemannianGD | RiemannianAdam` — a structural union, not a Protocol. Adding a third optimizer means extending that union *and* adding an `isinstance` branch in `optimize()`.

`train_basis` is generic over basis type via JAX pytree flatten/unflatten. The convention: each registered basis pytree must have leaves ordered as `tuple(tensors) + tuple(inv_tensors)`, with everything else (m, n, code, inv_code, counts) in aux data. New basis types must follow this convention.

## Dev workflow

```bash
# Install (Python 3.11+; conda env at /opt/conda/envs/pdft on the dev box)
pip install -e ".[dev]"

# Run tests
pytest                                    # full suite
pytest --cov=pdft --cov-fail-under=90     # CI gate
pytest tests/test_parity_*.py             # parity-only
pytest tests/completion                   # the completion subpackage (no goldens; property tests)

# Lint (CI fails if this is dirty — check before pushing)
ruff check src tests
ruff format src tests

# Run examples
python examples/basis_demo.py
python examples/optimizer_benchmark.py
python examples/mera_demo.py
python examples/completion_demo.py      # train through the solver, convert to QFTBasis, save

# Regenerate Julia goldens (requires Julia 1.10+)
make goldens
```

After regenerating goldens: also update `__upstream_ref__` in `src/pdft/__init__.py` to match `manifest.json`'s `upstream_sha`.

## Tests and parity

Three layers (per spec section 7):

1. **Parity tests** (`tests/test_parity_*.py`) — load committed `.npz` / `.json` goldens from `reference/goldens/` and assert Python matches Julia. These are the load-bearing correctness tests.
2. **Property tests** (`tests/test_<module>.py`) — math-invariant checks (unitarity preserved, round-trip identity, monotone descent, …). Don't depend on Julia.
3. **Smoke / integration** (`tests/test_smoke.py`, `tests/test_training_integration.py`).

Coverage gate is `--cov-fail-under=90`. Don't add tests that reduce per-module coverage below the line; if a new module legitimately needs more code, also add the property tests for it.

`tests/conftest.py` enables x64. Don't import jax in a test before pdft unless the test also calls `jax.config.update("jax_enable_x64", True)`.

## When to break parity

The default answer is "never". The only acceptable cases:

- **Float-precision noise**: test tolerances are documented per case; if you cross a tolerance boundary, investigate before relaxing.
- **Tie-breaking divergences** (e.g. argpartition vs partialsortperm): document and assert downstream behavior matches instead.

If you find another mismatch:
1. Diagnose with `reference/julia/` (write a small Julia script to dump intermediate values).
2. Compare to Python at the same intermediate point.
3. **The bug is in Python**, almost always.
4. If the bug is in upstream Julia, file an issue there and document the divergence in this file.

## CI gotchas

- Ruff is strict (`F401`, `F841`, `E741`, `E731` all errors). Auto-`--fix` handles most.
- Coverage gate runs only on the actual test suite, not examples; don't rely on examples for coverage.
- The `verify-upstream-pin.yml` workflow runs only on PRs touching `reference/`. It uses the GitHub API to confirm the pinned sha exists in upstream — don't push a sha that's only in a fork.
- The matrix is 3.11 / 3.12 / 3.13. JAX dropped 3.10 in 0.10.0; do not lower the floor.

## What NOT to do

- **Don't add `optax`** as a dependency. The optimizer logic is hand-rolled to match Julia's exact moment-update math. `optax`'s defaults and FP order will diverge from Julia.
- **Don't switch from `jnp.einsum_path("greedy")` to `"optimal"`.** "Optimal" is exponential in tensor count and hangs on the 3×3 QFT (12 tensors).
- **Don't add explicit JIT to `train_basis`.** It calls a basis-typed loss closure with Python-list pytrees; JIT decisions are best left to inner functions where the static-vs-leaf split is clearer.
- **Don't introduce backwards-compat shims** for the JSON schema. We're at v0.1.0; if the schema changes, bump the version and regenerate goldens.
- **Don't add ML scaffolding** (no DataLoader, no Trainer-like classes, no Lightning). Upstream is one-target-image-at-a-time and we mirror that. Batched training is open work in #2.
- **Don't move the paper's scripts, data or results into this repo.** `pdft.completion` is the library; the experiments live in pdft-completion. Anything two of that repo's scripts would share belongs here, with explicit paths, no `ROOT`.
- **Don't run examples in CI.** They write to `out/` (gitignored) and are not coverage-relevant.

## When making changes

1. Make the change.
2. Run `ruff check src tests` and `ruff format src tests`.
3. Run `pytest --cov=pdft --cov-fail-under=90`.
4. If you touched a circuit, basis, optimizer, or io_json module: also run the relevant `tests/test_parity_*.py` to confirm Julia-parity hasn't drifted.
5. If you changed the upstream pin: update both `reference/julia/generate_goldens.jl` (`UPSTREAM_SHA`) and `src/pdft/__init__.py` (`__upstream_ref__`), regenerate goldens, and verify all parity tests still pass.
6. Push and watch the GitHub Actions matrix — three Python versions must all be green before merging.
