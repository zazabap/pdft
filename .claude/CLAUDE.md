# CLAUDE.md

Project-specific guidance for Claude Code working in this repo. Read this before making non-trivial changes — several conventions are *not* discoverable from the code without painful debugging, and at least one is documented here only because the original Python port spent hours rediscovering it.

## What this is

`pdft` is a faithful Python port of [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl) using JAX. Goal: a Python user can reproduce Julia results bit-for-bit (or within documented tolerances) on the same input. **Behavior parity with Julia is the primary correctness criterion** — never sacrifice it for Pythonic-ness.

- Upstream has four bases: QFT, EntangledQFT, TEBD and MERA. `RichBasis`, `RealRichBasis`, `DCT4Basis`, `BlockedBasis`, the `"u4"` parametrizations, `freeze_as_blocked` and `pdft.coherence` exist only here. For those there is no Julia behaviour to match and no golden; their property tests are the specification.
- Image completion (`pdft.tasks.completion`, `train_basis_steps`, the parameter views, the flat manifold) is not in upstream either. Its reference is the inpainting paper's own code (arXiv:2609.17298, the private `pdft-completion` repository): `reference/completion/golden.npz` is a run of it and `tests/parity/test_completion.py` holds the package to that run. Section 14 says what the golden can and cannot pin.
- Design spec: `docs/superpowers/specs/2026-04-24-pdft-migration-design.md` (referred to below as "the spec"; the file is not in this repository)
- Roadmap: GitHub issue #1
- Reliability hardening backlog: GitHub issue #2
- Upstream is pinned to `nzy1997/ParametricDFT.jl@a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef`. The pin lives in two places that must stay in sync: `reference/julia/generate_goldens.jl` (`UPSTREAM_SHA`) and `src/pdft/__init__.py` (`__upstream_ref__`).

## Critical conventions (do not break)

These are the non-obvious invariants that took the original port multiple iterations to find. Violating any of them silently breaks parity with Julia.

### 1. Wirtinger gradient conjugation

**JAX and Julia's Zygote return conjugate gradients** for real-valued functions of complex inputs: `jax.grad` gives `∂f/∂x − i ∂f/∂y` (twice `∂f/∂z`; for `|z|²` at `1+2i` it returns `2−4i`), Zygote gives `∂f/∂x + i ∂f/∂y` (twice `∂f/∂z̄`, the direction of steepest ascent). `optimize()` in `src/pdft/optimizers/loop.py` conjugates `raw_grads` immediately after `grad_fn(...)`:

```python
raw_grads = grad_fn(state.current_tensors)
raw_grads = [jnp.conj(g) for g in raw_grads]   # MUST stay
```

Without this line, GD trajectories drift ~10% over 50 steps and Adam outright diverges. The fused batched step in `training/adam_step.py` applies the same conjugation, right after its own `value_and_grad`. Those are the only two places a gradient is turned into an update (the trainers and `fit_to_dct` hand their gradient function to `optimize`); any new optimizer or training step must do the same, or define a custom JAX `vjp` that matches Julia's convention.

### 2. Yao little-endian qubit ordering

`pic.reshape((2,)*(m+n))` gives axes in big-endian bit order, but Julia's Yao treats qubit 1 as the LSB. `circuit.builder._axis_of_qubit` is where the applier reverses the qubit-to-axis mapping within each register:

```python
if 1 <= q <= m:
    return m - q                # axis 0 ↔ qubit m, axis m-1 ↔ qubit 1
if m + 1 <= q <= m + n:
    return m + (m + n - q)      # axis m ↔ qubit m+n, axis m+n-1 ↔ qubit m+1
raise ValueError(...)
```

The applier (`_walk`) reads every gate's axes through it, forward and inverse. Four other places assume the same layout without calling it and have to change with it: `BlockCode` (which axes index the blocks), `freeze_as_blocked` (which qubits index the blocks), `bit_reverse`, and the DCT-IV emitter's `Q()`. Don't undo this without first making the `ft_mat` parity tests fail in a controlled way.

### 3. Compact 2×2 CP tensor representation

A controlled-phase gate is `diag(1, 1, 1, exp(iφ))` as a 4×4 matrix, but Yao's `yao2einsum` emits the **compact 2×2 form**:

```python
controlled_phase_diag(φ) = [[1, 1], [1, exp(iφ)]]
```

The gate is diagonal, so each wire's value is unchanged and only a multiplicative phase is injected: the applier broadcasts the 2×2 tensor onto the control and target axes and multiplies, transposing it when the control sits on the higher axis. Every gate of kind `CP` uses this representation; never substitute the 4×4 form for one even though it would be mathematically equivalent — Julia goldens won't match. The dense form exists as its own gate kind (`U4`: Rich, RealRich, DCT4, and the `"u4"` parametrization of TEBD and MERA), a different parametrization on a different manifold, chosen on purpose.

### 4. Hadamard-first tensor sort

Julia's `qft_code` does `perm_vec = sortperm(tn.tensors, by=x -> !(x ≈ mat(H)))` — Hadamards come first, everything else after, in emission order. Python's `circuit.builder.compile_program` applies the same sort (`_hadamard_first_perm`), once, from the initial tensor values. A basis stores its tensors in that order; the `Program` keeps the gates in the order they act and `Program.slot` maps each gate to its stored tensor, so the sort never has to be redone or guessed (`Program.sorted_steps` gives `(kind, qubits)` per stored tensor). **JSON serialization assumes this order**; cross-language interop breaks if you change it.

### 5. JAX x64 mode at import time

`src/pdft/__init__.py` calls `jax.config.update("jax_enable_x64", True)` before any other import. `tests/conftest.py` does the same. Without x64, JAX uses complex64 / float32 and parity tolerances become unreachable. Any new test file that uses `jax.numpy` directly without importing `pdft` first must include the same call (or import pdft to force it).

### 6. Column-major iteration for cross-language hashes

`io.serialize.basis_hash` and `tasks.compression` use **Julia 1-based column-major** flat indices. NumPy is row-major by default. Use `array.flatten(order="F")` or `np.unravel_index(..., order="F")`. Never use plain `array.flatten()` for serialization.

### 7. Julia-compatible float formatting

Python's `repr(5e-7)` gives `"5e-07"`; Julia's `string(5e-7)` gives `"5.0e-7"`. `io.serialize.format_float_julia_like` post-processes `repr()` to match Julia. Use it (not `repr` or `str`) for any value that participates in a cross-language hash or JSON byte-comparison.

### 8. L1-cusp horizon for trajectory parity

GD on L1 loss agrees with Julia to about `1e-14` for ~50 steps (the parity tests assert `atol=1e-10`; it is not equal to the bit) and stays within `atol=1e-3` over 200 steps. Beyond that, FP accumulation eventually pushes Python and Julia across the same loss cusp at slightly different alphas in Armijo line search. Both still converge to the same loss basin. Don't tighten `tests/parity/test_long_run.py` to `atol=1e-10` over 200 steps — this is mathematically expected on non-smooth losses, not a bug. Smooth losses (MSE without truncation) don't have this problem.

### 9. The upstream phase extractors classify by tensor values

`get_*_gate_indices` (`select_last_n_cp_indices` under Julia's names) walks the tensor list and tags any 2×2 tensor whose four entries have unit-modulus magnitudes (within `atol=0.15`) as a CP gate, then returns the LAST `n_gates` such positions. After training, individual entries can drift slightly off the unit circle; the moderate tolerance accommodates that. If you tighten the tolerance, do so in tandem with a regression test on a trained basis.

These mirror upstream and are kept for parity with it, limits included: "the last `n` CP tensors" is the entangle layer only for `entangle_position="back"`, and a `"u4"` parametrization has no compact-CP tensors to find. New code should not use them. The program has the exact answer (§12): the entangle layer is `tensor_indices(kind="CP", register="both")` wherever it sits, and the ring or layer gates of TEBD and MERA are the two-qubit gates, `kind="CP"` or `kind="U4"` by parametrization. For a `BlockedBasis` go through `program_of(basis)`. The other value-based classifier that stays is `manifolds.classify_manifold`, which picks each tensor's manifold as upstream does; a test asserts it agrees with the gate kind at the initial tensors of every basis.

### 10. One applier, and every basis keeps its program

There is one arithmetic by which a circuit reaches an image: a family emits a gate list (`<family>_gates`), `compile_program` turns it into a `Program` (structure only, hashable) plus the tensors in stored order, and `CircuitCode(program, inverse, slices)` applies it one gate at a time through the jitted `_walk`. `CircuitCode` compares and hashes by its program, so two bases with the same circuit share one compiled applier and have equal pytree structures. The inverse is the same walk backwards with each gate's legs swapped (the transpose); the caller conjugates the tensors, as Julia's `inverse_code(conj.(tensors)...)` does. The entry points onto that walk are `apply_circuit` and `contract_circuit` (one image; §13), `apply_program` (leading batch axes, precision taken from the image), and `BlockCode`, which maps an inner code over the blocks of a `BlockedBasis`.

`CircuitBasis` (`bases/core.py`) stores `program` next to `tensors` and derives the rest for every family: `code`, `inv_code`, the transforms, the parameter count and the pytree. A new family is its emitter plus `emit = staticmethod(<family>_gates)` (or a constructor that calls `_init` when it has options of its own). A `code` passed to a constructor defines the circuit: when it is a `CircuitCode` the basis takes its program from it, and one passed alone (as `code` or as `inv_code`) brings its counterpart, the same code in the other direction, so a basis rebuilt with another instance's tensors and codes stays consistent. Do not add a second applier, a per-family `forward_transform`, a per-family pytree registration, or code that recovers the gate sequence from tensor values: read `basis.program`.

The default arithmetic (`tensordot` with `precision="highest"`) is bit-identical to what the package computed before the gate-centred refactor on a CPU, in both precisions, and on a GPU in double precision. On a GPU in single precision it is not, on purpose: without `precision="highest"` XLA runs a complex64 contraction in TF32, and the transforms and losses of Rich, RealRich and Blocked bases were off by about `1e-3`; they are now within `3e-7` of the double-precision result. `slices=True` is the same operator in different arithmetic (measured in double precision: on a GPU from 1.2 to 2.5 times faster for a 3+3-qubit QFT up to 10 times for 10+10 qubits, and 2 to 6 times for a gradient; on a CPU about even for large circuits and up to twice slower for small ones) and is opt-in because it changes the low bits. To use it, construct the basis with `code=dataclasses.replace(basis.code, slices=True)` (for a `BlockedBasis`, construct its inner basis so).

### 11. `QFTBasis` is the DFT of the bit-reversed image, with `e^{+2 pi i kx/N}`

Yao numbers qubits from the least significant bit and the QFT circuit has no final swap layer, so at its initial tensors

```python
QFTBasis(m, n).forward_transform(bit_reverse(x)) == np.fft.ifft2(x, norm="ortho")
```

where `circuit.bit_reverse` reverses the bits of the row index and of the column index (`Pi x Pi`, an involution). Two things to keep straight: the sign is numpy's *inverse* transform, and the frame is a fixed pixel permutation of the image. Neither is a bug and neither may be "fixed" in the basis: the Julia goldens encode both. Sparsity does not care about a pixel permutation. Anything defined on the pixels does (a sampling mask drawn from a seed, a figure): apply `bit_reverse` to the image going in and to the reconstruction coming out. `tests/circuit/test_bit_reverse.py` pins the identity.

### 12. Model variants are views of a basis, not new representations

A model that trains part of a circuit, or trains it through other parameters, is expressed on the basis that already exists:

- **Freeze by gate kind or register:** `basis.program.tensor_indices(kind="H")` (also `register="row" | "column" | "both"`) is a `frozen_indices` list. `program_of(basis)` reaches through a `BlockedBasis`.
- **Phase-only training:** `cp_phases(basis)` reads the controlled-phase angles as a real array and `with_cp_phases(basis, angles)` puts them back; `cp_diagonals` / `with_cp_diagonals` do the same for all four phases of each gate. Both are traceable, so a loss differentiates through them.
- **Parameter views:** a `ParameterView` (`bases/core.py`) is what a trainer moves: `read(basis)` gives a list of arrays, `write(basis, params)` a basis holding them, `manifolds(params)` the manifold of each or `None` for "classify by value as always". `tensors_view` is the tensors on their manifolds, the default everywhere; `cp_phases_view` and `cp_diagonals_view` are the angles as free real numbers on `EuclideanManifold`, read in double precision whatever the tensors' precision. On a flat view the package's one Adam update (`optimizers/adam.py::_adam_update`) is plain Adam as Kingma and Ba state it; there is no second update function, and there must not be one. Lowercase names on purpose: a constant `CP_PHASES` next to `cp_phases` collided on case-insensitive filesystems in the docs build.
- **Coherence:** `coherence.diagonal_tensor_indices` reads the gate kind from the program; `certify_flat_modulus` certifies `mu == 1` for a frozen configuration (it checks that the basis is flat now and that every trainable tensor is a `CP` gate, not that there is one Hadamard per wire, which holds for every basis here) and `sampled_flat_modulus` measures it.

Do not add a second way to hold a circuit's parameters (an angle vector with its own applier, a gate dict) and a bridge between the two. A new parametrization is a new view.

### 13. Two ways a transform reaches its circuit, on purpose

`apply_circuit` checks that the image is `(2**m, 2**n)` and casts it to complex128. `contract_circuit` does neither: any image with the right number of elements is reshaped, and the result has whatever precision the tensors and the image promote to.

- QFT, EntangledQFT, TEBD, MERA and DCT4 transforms use `apply_circuit`.
- Rich, RealRich and Blocked transforms, and `loss_function`, use `contract_circuit`. With single-precision tensors and a single-precision image they therefore compute in single precision: the transform returns complex64, and `loss_function` a float32 loss with complex64 gradients. A double-precision image promotes the result. Training follows the optimizer, in both trainers: `RiemannianGD` returns tensors in the precision they came in with, `RiemannianAdam` returns complex128 whatever went in (a tensor frozen during the run comes back as it went in). `train_basis_batched` also casts every image to complex128.

This asymmetry is how the package behaved before the refactor (`BasisTransforms._apply` now names it). It is not a convention worth defending, but unifying it changes results wherever single-precision tensors meet a single-precision image (`loss_function`, `train_basis`, the three transforms). It was once "cleaned up" by accident and no test noticed, because every test used double-precision tensors. `tests/bases/test_contracts.py` now pins both halves. Change it only as a decision, with the change declared.

### 14. Tasks work in the basis's own frame, and the step trainer owns the mask

A task (`pdft/tasks/`) takes a basis and data and reaches the circuit only through `forward_transform` and `inverse_transform`, so it runs on every basis, blocked ones included. `compress` / `recover` mirror upstream's `compression.jl`; `complete` (iterative hard thresholding from the observed pixels) and `completion_loss` are new. Four things to keep straight:

- **Frame.** Everything in `tasks` is in the basis's own frame, as `compress` always was. A mask lives on the pixels, so for a QFT-topology basis the caller bit-reverses the observation and the mask going in and the reconstruction coming out (§11). The frame is a property of the basis, not of the task: `DCT4Basis` and `BlockedBasis` lose several dB if bit-reversed. Do not put `bit_reverse` inside a task.
- **The mask belongs to `train_basis_steps`.** `train_basis_batched` mirrors Julia, where an example is an image and the loss a function of it; it never sees a mask and must not grow one. `train_basis_steps` (`training/steps.py`) runs a fixed number of Adam steps, each on a fresh batch under a fresh mask, drawn in the inpainting paper's order from one `np.random.default_rng(seed)`: the batch (`choice`, without replacement) and then its masks (`random(batch.shape) < rate`). That order is what makes a seed name the same batches and masks as in the paper's code; keep it. Its `frame=` maps batch and mask into the basis's frame. Below the loop nothing is separate: the fused step of `training/adam_step.py`, the one Adam update, the views.
- **The tie band.** At the Fourier point the coefficients of a real image come in conjugate pairs of equal magnitude; a cut that falls inside a pair (an even `k`, and at small sizes an odd `k` too, when a Nyquist bin enters the top `k`) was decided by rounding and changed the gradient by 1e-3 with the device and the arithmetic. `topk_truncate(x, k, rtol=)` counts magnitudes within `rtol` of the k-th largest as tied and settles ties by position; `complete` passes `sqrt(eps)`. The default `rtol=0` is upstream's rule, computed by upstream's two masks on the magnitudes as they are (a float band would round integers above 2**53); `MSELoss` and compression stay on it.
- **The barrier.** `complete` takes its real part as `jnp.real(jax.lax.optimization_barrier(z))`. With JAX 0.11.2 XLA carries a real or imaginary part back through the complex gates before it and compiling takes exponentially long in the depth of the circuit: 104 s against 0.1 s for a 16x16 `QFTBasis`, and a 30-minute CI job. The barrier changes no number; a test pins it because nothing numerical would notice its removal. The existing losses and trainers take magnitudes and are unaffected. Any new code that jits `jnp.real` or `jnp.imag` of a transform's output needs the same barrier.

The golden from the paper's code was generated with settings where no threshold in the run comes near a tie (smallest relative gap 3.7e-4): the paper's threshold keeps every magnitude that reaches the k-th, so `k + 1` at an exact tie, and the two codes agree only away from ties. The paper's own run differs between a CPU and a GPU by several percent when a tie is hit, so a golden from new settings must be checked that way first (`reference/completion/generate_golden.py` says how). With the same seed, odd `k` and the paper's settings at 128x128, `train_basis_steps` reproduces the paper's loss history to 2.5e-11 over 100 steps; the paper's Table 1 on DIV2K reproduces to 0.01 dB.

## Repo layout

```
src/pdft/
├── manifolds.py            Mathematical core (UnitaryManifold, PhaseManifold, EuclideanManifold, batched ops)
├── loss.py                 L1Norm, MSELoss, topk_truncate (with the optional tie band), loss_function (public)
├── profiling.py            Cross-cutting profiling helpers
├── bases/
│   ├── core.py             CircuitBasis (program + tensors -> transforms, pytree), AbstractSparseBasis, bases_allclose, the parameter views
│   ├── base.py             QFT, EntangledQFT, TEBD, MERA, DCT4 basis classes: each names its emitter
│   ├── circuit/            Gate emitters per family (qft, entangled_qft, tebd, mera, dct4, rich, real_rich) + freeze_as_blocked
│   └── block/              BlockedBasis (Rich/RealRich re-exported from circuit for back-compat)
├── circuit/                builder.py: Gate, Program, compile_program, CircuitCode (the one applier), gate constructors
├── coherence.py            Mutual coherence of a basis and the flat-modulus certificate
├── optimizers/             core, gd (RiemannianGD + Armijo), adam (RiemannianAdam + the one Adam update both drivers use), loop
├── training/               schedules, single (train_basis), batched (one epoch loop; Adam and GD supply the batch and the step), steps (train_basis_steps: a fresh batch and mask per step), adam_step (the fused step both dataset trainers use), eval_loop, result (TrainingResult)
├── tasks/                  compression (mirror of upstream compression.jl, with its JSON form), completion (complete, completion_loss)
├── io/                     serialize (basis JSON)
└── viz/                    loss (matplotlib loss plots), circuit (the gate sequence a basis keeps), _figure (shared import guard + save)

reference/julia/            Julia harness — needed only to regenerate goldens
reference/goldens/          Committed .npz + .json files (<200 KB total)
reference/completion/       The completion golden and the script that generated it in the paper's repository
examples/                   4 runnable demos, a few seconds each
tests/                      pytest; mirrors src/pdft/ layout (tests/bases/, tests/optimizers/, ...)
docs/                       Sphinx site (conf.py, index.md, api/*.rst, paper.md); deployed to GitHub Pages by docs.yml
```

Benchmarks live in a separate repo: https://github.com/zazabap/pdft-benchmarks (split from this repo at pdft v0.2.0; pinned via its `pyproject.toml`).

`DCTBasis` was removed during the modular-src refactor (PR #11). For a cosine transform use `DCT4Basis`; `RichBasis` / `RealRichBasis` are the general parametric alternatives.

`pdft/__init__.py` re-exports a small set of most-used names (basis classes,
trainer, optimizer, loss). Less-public names live in their subpackages —
e.g. `pdft.io.save_basis`, `pdft.tasks.compress`, `pdft.manifolds.UnitaryManifold`,
`pdft.profiling.profile_training`, `pdft.bases.circuit.qft_code`.

`AbstractRiemannianOptimizer` is `RiemannianGD | RiemannianAdam` — a structural union, not a Protocol. Adding a third optimizer means extending that union, adding an `isinstance` branch in `optimize()`, and teaching `training/batched.py` about it (`_resolve_optimizer` and the step the epoch loop takes).

`train_basis` is generic over basis type via JAX pytree flatten/unflatten. The convention: the leaves of a basis pytree are its `tensors`, in order, and everything else (m, n, program, code, inv_code, counts) is aux data. There is no separate inverse tensor list: `inv_tensors` is the same list, and the inverse applies `conj(tensors)` through `inv_code`. `CircuitBasis` registers every subclass this way; `BlockedBasis` delegates to its inner basis. New basis types must follow this convention. `bases.with_tensors(basis, tensors)` is the one way to get a copy of a basis holding other tensors (the trainers, `freeze_as_blocked` and the parameter views all use it); don't re-derive it with `tree_flatten` / `tree_unflatten` or by calling the constructor again.

## Dev workflow

```bash
# Install (Python 3.11+)
pip install -e ".[dev]"

# Run tests
pytest                                    # full suite
pytest --cov=pdft --cov-fail-under=90     # CI gate
pytest tests/parity                       # parity-only (one more golden-backed test is tests/bases/test_phase_extraction.py; tests/parity/test_completion.py is against the paper's code, not Julia)

# Lint
ruff check src tests                      # the CI gate: fails the build if dirty
ruff format <the files you touched>       # CI does not run it; four test files from before are unformatted, leave them

# Run examples
python examples/basis_demo.py
python examples/optimizer_benchmark.py
python examples/mera_demo.py

# Build the docs site (Sphinx + sphinx-book-theme; same -W command CI runs)
pip install -e ".[docs]"
make docs                                 # -> docs/_build/html

# Regenerate Julia goldens (requires Julia 1.10+)
make goldens
```

After regenerating goldens: also update `__upstream_ref__` in `src/pdft/__init__.py` to match `manifest.json`'s `upstream_sha`.

## Tests and parity

Three layers (per spec section 7):

1. **Parity tests** (`tests/parity/test_*.py`) — load committed `.npz` / `.json` goldens from `reference/goldens/` and assert Python matches Julia. These are the load-bearing correctness tests. `test_completion.py` is the one exception: its golden is `reference/completion/golden.npz`, a run of the inpainting paper's code (§14).
2. **Property tests** (`tests/test_<module>.py` and `tests/<subpackage>/`) — math-invariant checks (unitarity preserved, round-trip identity, monotone descent, …). Don't depend on Julia. The test tree mirrors `src/`: `tests/circuit/` covers `circuit/builder.py` (`test_gates.py`, `test_program.py`, `test_applier.py`, `test_bit_reverse.py`, and three regression files that are older: `test_cry_gate.py`, `test_h_inverse_leg_swap.py`, `test_high_qubit_circuit.py`), `tests/bases/circuit/` the family emitters, `tests/bases/` the shared basis machinery. Name a test file after what it tests, not after the change that added it. `tests/circuit/test_applier.py` checks the gate walk against a single einsum, a second statement of what a gate list means, for every registered circuit the einsum can express (no controlled-rotation gate, not blocked) at random tensors. `tests/bases/test_contracts.py` runs what every basis has in common over one registry of basis configurations (`BASES` in `tests/helpers.py`), including the precision and shape each family's transforms accept and return (§13); add a new basis to that registry. `tests/helpers.py` also holds the random inputs several files share, and `PlainAdam`, the numpy oracle for the update on flat parameters: use them before writing another. `tests/tasks/` covers `pdft/tasks/`; `tests/training/test_steps.py` the step trainer.
3. **Smoke / integration** (`tests/test_smoke.py`, `tests/training/test_integration.py`).

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

## Known defects

Found while refactoring and left as they are, because fixing each one changes behaviour:

- `DCT4Basis` is not exactly real: its sign gate is stored as `exp(i*pi)`, whose imaginary part is `1.2e-16`, and the optimisers amplify it (about `1e-2` in the tensors after twenty Adam steps on real images). `RealRichBasis`, whose tensors are exactly real, stays real.
- `TEBDBasis` with a one-qubit register (`m == 1` or `n == 1`) constructs and fails at its first transform: the ring gate lands on `(q, q)`.
- `==` on two bases raises `ValueError` as soon as it reaches two distinct arrays (the dataclass comparison reaches lists of arrays), and is `True` only when they share every tensor object, as two `QFTBasis(1, 1)` do. Use `bases_allclose`.
- JSON serialization is for `QFTBasis` only, and `basis_to_dict` does not refuse another basis: it labels it `"QFTBasis"`. Upstream serialises all four of its bases, and has an `entangle_position=:middle` that is not ported.
- `classify_manifold` goes by tensor values, as upstream does, so a unitary gate that has drifted too far is trained as a phase tensor. The test is `jnp.allclose(T T†, I, atol=1e-6)`, which adds a relative `1e-5`; upstream's `isapprox(t * t', I, atol=1e-6)` has no relative part, so the two disagree for a drift between those.
- On a GPU with single-precision tensors that same test runs in TF32 and misses unitarity by `2e-4`: Hadamards and dense gates are classified as phase tensors, a few steps of either optimizer then leave them far from unitary (round-trip error of order 10), and `DCT4Basis` fails to stack. Train in double precision on a GPU. The suite passes on a GPU without noticing. The flat views (`cp_phases_view`, `cp_diagonals_view`) name their manifold instead of classifying by value and are not affected.
- `topk_truncate` with `rtol > 0` compares through a float band, so integer inputs above `2**53` are rounded there; at `rtol=0` they are compared exactly, as on upstream.
- `complete` with a `QFTBasis` runs in double precision whatever the image's precision, because the QFT transforms cast (§13); with single-precision Rich tensors and a float32 image it runs in single precision.
- `certify_flat_modulus` does not check the proposition's "one Hadamard per wire"; a hand-built circuit with two on a wire is certified although `mu` can reach 2.
- Under `RiemannianGD` a frozen phase tensor can come back one rounding (`1e-16`) away from where it started; under Adam it is bit-identical.
- `pdft.coherence` at the package root is the function, which shadows the module: `from pdft.coherence import ...` works, `import pdft.coherence as c` gives the function.
- A code passed alone (`code=` or `inv_code=`) brings its counterpart only when it is a `CircuitCode`; any other callable is paired with the default circuit's.
- `profile_training` compiles its step twice: the record it tags as the first `"warm"` step includes a second compilation (about a thousand times the later steps), so drop it too when reading timings.
- `is_flat_modulus` compares with `jnp.allclose`, so `atol` is not the whole tolerance (a relative `1e-5` is added). `flat_modulus_deviation` gives the number.

## CI gotchas

- Ruff is strict (`F401`, `F841`, `E741`, `E731` all errors). `--fix` removes unused imports; the other three need a hand.
- Coverage gate runs only on the actual test suite, not examples; don't rely on examples for coverage.
- The `verify-upstream-pin.yml` workflow runs only on PRs touching `reference/` (or the workflow file itself). It uses the GitHub API to confirm the pinned sha exists in upstream — don't push a sha that's only in a fork.
- The matrix is 3.11 / 3.12 / 3.13. JAX requires Python 3.11 from 0.7.0; do not lower the floor.
- The matrix installs whatever JAX is current, and the jobs can get different versions (one run had 0.10.2 on 3.11 and 0.11.2 on the others). A green run that took 30 minutes on two jobs and 4 on the third was the real-part compile blow-up of §14. Look at job durations, not only at the ticks, and test under the newest JAX before pushing a change to anything that jits a transform.
- `docs.yml` deploys with `actions/deploy-pages`, which needs the repository's Pages source set to **GitHub Actions** (Settings → Pages, one-time). Until that is set the `deploy` job fails with a 404 while `build` stays green.

## What NOT to do

- **Don't add `optax`** as a dependency. The optimizer logic is hand-rolled to match Julia's exact moment-update math. `optax`'s defaults and FP order will diverge from Julia.
- **Don't bring back a whole-circuit einsum in the package.** Circuits are applied one gate at a time: no contraction path to search, no 52-label limit. The einsum form survives only as the oracle in `tests/circuit/test_applier.py`, which uses the `"greedy"` path; `"optimal"` is exponential in tensor count and hangs on the 3×3 QFT (12 tensors).
- **Don't add explicit JIT to `train_basis`.** It calls a basis-typed loss closure with Python-list pytrees; JIT decisions are best left to inner functions where the static-vs-leaf split is clearer.
- **Don't introduce backwards-compat shims** for the JSON schema. If the schema changes, bump its version and regenerate goldens.
- **Don't add ML scaffolding** (no DataLoader, no Trainer-like classes, no Lightning). `train_basis` is `optimize` on one target image, which is what the goldens harness runs (`reference/julia/generate_goldens.jl` calls `optimize!` directly; upstream has no one-image trainer), `train_basis_batched` is upstream's `train_basis` / `_train_basis_core`, and `train_basis_steps` is the inpainting paper's `adam_loop`; all three are plain functions.
- **Don't add a second optimizer for "plain" Adam, and don't add a second training loop for a new task.** Plain Adam is the Riemannian update on a flat view (§12); a task supplies an objective `(basis, images, masks) -> scalar` to `train_basis_steps`. The old completion stack (#34–#44) had its own kernel, bridge, Adam and loop, about 2,800 lines; #61 replaced it with about 600, and that is the shape to keep.
- **Don't jit `jnp.real` or `jnp.imag` of a transform's output without `jax.lax.optimization_barrier`** (§14).
- **Don't run examples in the test CI.** They write to `out/` (gitignored) and are not coverage-relevant. The one place they do run is the docs workflow (`docs.yml`): sphinx-gallery executes `examples/*.py` to render the example gallery, under `-W`, so a broken example fails the docs build rather than the test matrix.

## When making changes

1. Make the change.
2. Run `ruff check src tests`, and `ruff format` on the files you touched.
3. Run `pytest --cov=pdft --cov-fail-under=90`.
4. If you touched a circuit, basis, optimizer, or io module: also run `tests/parity` to confirm Julia-parity hasn't drifted. For a change meant to alter no numbers, compare old and new on the same machine before saying so: bytes, not `allclose`, on the CPU (on a GPU two runs of the same code need not agree to the bit), and with single-precision tensors as well as double (§13).
5. If you changed the upstream pin: update both `reference/julia/generate_goldens.jl` (`UPSTREAM_SHA`) and `src/pdft/__init__.py` (`__upstream_ref__`), regenerate goldens, and verify all parity tests still pass.
6. Push and watch the GitHub Actions matrix — three Python versions must all be green before merging.
