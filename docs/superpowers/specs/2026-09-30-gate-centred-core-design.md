# Gate-centred core: one circuit definition, one applier

- **Status:** proposal, not accepted. Nothing in `src/` changes with this document.
- **Date:** 2026-09-30
- **Baseline:** `main` at `102f5b6` (v0.2.4). Line references below are to that commit.
- **Prompted by:** the review of #35, where the image-completion port added a second gate-level implementation of the QFT circuit next to the one the package already has.
- **Ordering constraint:** this refactor lands before any completion module is added. The completion stack (#35 to #44) is re-cut on top of it afterwards.

## 1. Summary

A circuit in this package is already a list of gates, and every basis already applies that list one gate at a time. But the gate list is thrown away after construction, so the rest of the package re-derives it, guesses it, or duplicates the code around it. The completion port then added a second applier for the same circuit because the first one lacked three capabilities (batch axes, single precision, speed at large sizes).

The proposal is to make the gate program the one source of truth:

1. **One program object** that a basis keeps, instead of two opaque closures.
2. **One applier** for every basis and for the completion operators, aware of batch axes and of the image's dtype.
3. **One basis implementation** shared by the seven circuit bases, which today repeat 60% of their lines.
4. **One copy of each gate emitter**, in particular the QFT skeleton, which exists three times.
5. **The hooks completion needs** (parameter views, freezing by gate kind, the pixel-frame adapter) added to the core, so the later stack has nothing to duplicate.

Julia parity is the constraint throughout: tensor values, the Hadamard-first tensor order, the JSON schema, the public class names and constructor signatures do not change.

## 2. What `main` looks like today

### 2.1 The live path is already gate by gate

Every basis builds a `list[Gate]` and calls `compile_circuit` (`circuit/builder.py:306`), which returns a jitted closure over `_stepped_apply` (`builder.py:213`). That function walks the program and applies one gate per step, for four kinds: `H` (any one-qubit tensor), `CP` (compact diagonal 2x2), `U4` (dense two-qubit) and `CRY`.

The single-einsum builder (`build_circuit_einsum`, `builder.py:99`) and its path cache (`circuit/cache.py`) are no longer on any production path. Nothing in `src/` calls them; three test files do (`test_legacy_builder.py`, `test_high_qubit_circuit.py`, `test_cache.py`), where they serve as an independent oracle for the stepped path. `CLAUDE.md` conventions 2 and 4 and the `einsum_path` rule still describe the einsum builder as the live one.

### 2.2 The program is not kept, so everything around it repeats

| Symptom | Where |
|---|---|
| Seven circuit basis classes plus `BlockedBasis`: 612 lines of code, of which 369 (60%) are the same line repeated in at least four classes (`image_size`, `num_parameters`, `inv_tensors`, `forward_transform`, `inverse_transform`, flatten, unflatten) | `bases/base.py`, `bases/circuit/rich.py`, `real_rich.py`, `bases/block/block.py` |
| The QFT gate loop written three times, differing only in the two-qubit gate it emits | `qft.py:45`, `rich.py:40`, `real_rich.py:47` |
| The "diagonal or dense two-qubit gate" factory written twice | `tebd.py:105`, `mera.py:80` |
| Three ways for a basis to call the applier | `builder.apply_circuit`, `loss._apply_circuit` (a copy without the cast and the shape check), the `ft_mat` / `ift_mat` wrappers |
| `freeze_as_blocked` rebuilds the gate list through a table keyed on basis type, and therefore supports three bases | `freeze.py:70` |
| The circuit plot cannot draw the topology: "the explicit gate sequence which we don't preserve in Basis" | `viz/circuit.py:69` |
| `QFTBasis.__init__` rebuilds both gate lists and both closures on every construction, including every pytree unflatten | `base.py:73`, `base.py:130` |
| Each basis instance owns its own jitted closures, so two instances of the same circuit compile twice | `compile_circuit` |

### 2.3 What the completion port duplicated

The port carried the QFT circuit as stacked gate parameters and applied it with its own kernel (`apply_gates`). That kernel and `QFTBasis.forward_transform` are the same operator up to a bit reversal and a conjugation: they agree to 1.6e-14 on a 512x512 image. Around that second representation the port needed a bridge between the two, a second coherence module, and three "families" that are `QFTBasis` under a different parameterisation or with different gates frozen.

By module size, roughly 750 to 900 of the port's 2,803 source lines exist to maintain the second representation (`transform`, `bridge`, `coherence`, `families/general`, `families/phases`, `families/shared`, and parts of `riemannian`, `training` and `solver`). This is an estimate from line counts, not a measured reduction. The solver, the memory-bounded solver, metrics, protocol, data splits, butterfly, transform learning and the per-image baselines are new under any design.

Two smaller overlaps with existing code: `solver.hard_k` against `loss.topk_truncate`, and the compression control `comp_loss` against `MSELoss(k)`.

## 3. Evidence that one applier can serve both

Measured on an RTX 5070 and a Ryzen 5 9600X, JAX 0.10.2, x64 enabled. "Explicit" means the gate is applied as a linear combination of the slices of the wires it touches, the arithmetic the completion kernel used, instead of `jnp.tensordot`. The change was installed by monkeypatching `_stepped_apply` for measurement; nothing was committed.

### 3.1 Tests

| | Result |
|---|---|
| `main`, unmodified | 280 passed, 2 skipped |
| `main`, every gate kind explicit | 280 passed, 2 skipped |

Every Julia parity test passes under the changed arithmetic. The numbers move at rounding level, inside every documented tolerance.

### 3.2 The completion workload (GPU)

| | `main` as is | one-qubit gates explicit | completion kernel |
|---|---|---|---|
| Forward transform, 512x512, float64 | 10.4 ms | 1.06 ms | 1.13 ms |
| One gradient through a 20-step solver, 256x256, float64 | 490 ms | 123 ms | 128 ms |
| One gradient through a 30-step solver, 512x512, float32 | not measured | 48 ms | 67 ms |
| Single-precision error against float64 (largest coefficient 256) | 0.5 | 7e-5 | 7e-5 |

The single-precision failure comes from XLA lowering float32 matmuls to TF32 on this card; under `jax.default_matmul_precision("highest")` the unmodified applier also reaches 7e-5. Explicit arithmetic has no matmul, so it needs no such context.

### 3.3 The cost, per basis

Steady-state time and first-call time (tracing plus compilation) of an L1 gradient, as is → explicit. The first-call columns are cold: the first time that graph was ever compiled on the machine. Section 3.4 separates cold from warm.

| Basis, size | GPU per gradient | GPU first call | CPU per gradient | CPU first call |
|---|---|---|---|---|
| `QFTBasis` 8x8 | 0.31 → 0.09 ms | 0.10 → 0.26 s | 0.016 → 0.024 ms | 0.09 → 0.17 s |
| `QFTBasis` 64x64 | 1.02 → 0.44 ms | 0.34 → 0.80 s | 0.56 → 0.80 ms | 0.27 → 0.74 s |
| `QFTBasis` 512x512 | 41.4 → 7.2 ms | 1.02 → 2.20 s | 49.9 → 41.8 ms | 0.68 → 1.51 s |
| `RichBasis` 8x8 | 0.52 → 0.15 ms | 0.09 → 0.79 s | 0.022 → 0.076 ms | 0.10 → 0.54 s |
| `RichBasis` 64x64 | 2.45 → 2.59 ms | 0.29 → 4.99 s | 1.08 → 1.61 ms | 0.23 → 3.83 s |
| `RichBasis` 512x512 | 89.0 → 16.5 ms | 0.83 → 16.45 s | not measured | not measured |

Reading:

- **One-qubit gates (`QFTBasis`):** a clear win on the GPU at every size (up to 5.7x per gradient, 10x on the forward transform at 512x512). On the CPU it is about 1.4x slower at small sizes and slightly faster at 512x512. Cold compilation takes about twice as long.
- **Dense two-qubit gates (`RichBasis`):** the explicit form is a 16-term sum per gate. It is 5x faster per gradient at 512x512 on the GPU, but cold compilation is up to 20x slower, and on the CPU at small sizes the gradient is 1.5 to 3.5x slower. The steady-state slowdown on the CPU is the part that caching cannot help; see open question Q1.
- The whole test suite takes 30 s instead of 11 s with every kind explicit and a cold cache, which is compilation of many small circuits.

### 3.4 Compilation: what it is, and what the existing cache recovers

Compilation time is proportional to the size of the traced graph. A circuit is unrolled gate by gate into one flat graph, and XLA compiles every operation in it, at roughly 0.6 to 1 ms per operation on this machine.

| L1 gradient at 512x512 | operations in the lowered graph | trace and lower (Python) | XLA compile, cold |
|---|---|---|---|
| `QFTBasis`, as is | 847 | 0.11 s | 0.82 s |
| `RichBasis`, as is | 556 | 0.11 s | 0.60 s |
| `QFTBasis`, explicit | 2,522 | 0.19 s | 1.95 s |
| `RichBasis`, explicit | 23,281 | 1.44 s | 14.46 s |

The explicit two-qubit form is slow to compile only because it emits about 40 times more operations.

**The package already caches compiled code on disk.** `pdft/__init__.py:33` turns on JAX's persistent compilation cache at import (`~/.cache/pdft/jax-compile-cache`, every compile stored, `PDFT_DISABLE_COMPILE_CACHE=1` to opt out). So the cold cost is paid once per machine per graph, not once per run. First call of the same gradient:

| | as is, no cache | as is, warm | explicit, no cache | explicit, warm |
|---|---|---|---|---|
| `QFTBasis` 512x512, GPU | 1.16 s | 0.30 s | 2.11 s | 0.31 s |
| `RichBasis` 512x512, GPU | 0.82 s | 0.26 s | 15.3 s | 2.2 s |
| `QFTBasis` 64x64, CPU | 0.27 s | 0.07 s | 0.77 s | 0.14 s |
| `RichBasis` 64x64, CPU | 0.22 s | 0.06 s | 3.73 s | 0.85 s |

What the cache does not recover:

- **Tracing.** The cache key is the lowered graph, so Python still traces and lowers the circuit in every process: 1.44 s of the 2.2 s warm figure for `RichBasis` at 512x512. Only a smaller graph reduces this.
- **Any change to the graph.** A new image size, batch size, dtype, loss or solver depth is a new key and a cold compile.
- **CI.** The workflows cache pip only, so every CI run is cold.
- **Steady-state time.** Caching has no effect on how fast the compiled code runs.

Without the disk cache a second instance of the same circuit in the same process recompiles from scratch (0.71 s as is, 15.9 s explicit, for `RichBasis` at 512x512), because each basis instance owns its own jitted closures. A program-keyed applier (section 4.2) shares compiled code and the trace across instances in a process regardless of the disk cache.

## 4. Design

Names are provisional.

### 4.1 `Program`: the circuit as data

```python
@dataclass(frozen=True)
class Program:
    """The structure of a circuit: no tensor values, hashable, usable as a static jit argument."""
    m: int
    n: int
    steps: tuple[tuple[str, tuple[int, ...]], ...]   # (kind, qubits) in temporal order
    slot: tuple[int, ...]                            # step i reads tensors[slot[i]]

def compile_program(gates: list[Gate], m: int, n: int) -> tuple[Program, list[Array]]:
    """The program and its initial tensors, in the Hadamard-first order."""
```

`slot` records the Hadamard-first permutation that `compile_circuit` computes today from the initial tensor values (`_hadamard_first_perm`). It is computed once, by the same rule, and stored. The tensor list a basis holds keeps its current order, so JSON and the goldens are untouched.

`Gate` stays as it is. `sorted_gate_program` becomes a view of `Program`.

### 4.2 One applier

```python
def apply_program(program: Program, tensors, x: Array, *, inverse: bool = False) -> Array:
    """Apply the program to ``x`` of shape ``(..., 2**m, 2**n)``."""
```

- **Batch axes.** Leading axes of `x` are carried through. `BlockedBasis` can then express its block axes as batch axes instead of wrapping the inner closure in a `vmap`.
- **Precision.** The working dtype follows the image: complex64 for float32 or complex64 input, complex128 otherwise. The existing `apply_circuit` keeps its cast to complex128, so nothing that exists today changes type.
- **Inverse.** `inverse=True` walks the reversed program with the transposed-leg convention and the caller conjugates the tensors, exactly as today (`inverse_code(conj.(tensors)...)` in Julia). A convenience `adjoint=True` may do both.
- **Compilation.** Jitted with the program static, so the trace and the compiled code are shared by every basis instance with the same program. This is an inner-function jit, consistent with the rule not to jit `train_basis` itself. The package's existing disk cache keeps working unchanged and carries compiled code across processes (section 3.4); restoring that directory in CI would remove the cold cost there.
- **Arithmetic.** One-qubit gates use the explicit two-slice form. `CP` keeps its broadcast multiply. `U4` and `CRY` are decided by Q1.
- **Back-compatibility.** `basis.code` and `basis.inv_code` remain, as thin closures over `apply_program` on the `(2,) * (m + n)` layout, so `loss_function`, `train_basis`, `train_basis_batched` and `BlockedBasis` keep their signatures.

### 4.3 One copy of each emitter

`circuit/library.py` (or the existing per-family files, thinned):

- `qft_gates(n_qubits, offset, two_qubit=...)`: the QFT skeleton once, parameterised by the two-qubit gate factory. Replaces `_qft_gates_1d`, `_rich_qft_gates_1d`, `_real_rich_qft_gates_1d`. The old names stay as one-line aliases, since `_qft_gates_1d` is in `qft.__all__`.
- `two_qubit_gate(kind, ctrl, tgt, phi)`: the diagonal-or-dense factory once. Replaces the copies in `tebd.py` and `mera.py`.
- The entangle layer, the TEBD rings, the MERA layers and the DCT-IV program move as they are.
- `get_*_gate_indices` and `extract_*_phases` keep their public names (they mirror Julia) over one implementation.

### 4.4 One basis implementation

```python
class CircuitBasis:
    m: int
    n: int
    tensors: list[Array]
    program: Program
    # image_size, num_parameters, inv_tensors, forward_transform, inverse_transform,
    # code, inv_code, and pytree registration are defined here once.

class QFTBasis(CircuitBasis):
    def __init__(self, m, n, tensors=None, code=None, inv_code=None):
        ...  # validate, emit gates, call the shared initialiser
```

Each concrete class keeps its name, constructor signature, extra fields (`n_entangle`, `n_row_gates`, ...), seeding rule and pytree leaf order. What it loses is the repeated body. `bases_allclose` and `AbstractSparseBasis` are unchanged. `BlockedBasis` stays a wrapper.

Consumers that re-derive or guess the program read it from the basis instead:

- `freeze_as_blocked` works for every circuit basis, not three.
- `viz.plot_circuit` draws the real topology.
- `coherence.diagonal_tensor_indices` can use the gate kind; the value-based `is_compact_cp` stays for the Julia-mirrored helpers.
- `classify_manifold` stays value-based (it mirrors Julia). A test asserts it agrees with the manifold each gate kind implies, on every basis at initialisation.

### 4.5 Removing what is left over

- `loss._apply_circuit` goes; `loss_function` keeps its signature.
- `build_circuit_einsum` and `circuit/cache.py` leave `src/` and become a test-only reference, which is how they are used already. Their names in `pdft.circuit.__all__` get a deprecation alias for one release, following the `_format_float_julia_like` precedent. GitHub code search finds no use of them in `pdft-benchmarks`; that should be confirmed on a clone before removal.
- `CLAUDE.md` conventions 2 and 4 and the `einsum_path` rule are rewritten to describe the live path.

### 4.6 The hooks completion needs

Added in this refactor so the completion stack does not reintroduce a second representation:

- **Parameter views.** The phases of the `CP` gates as an array and back (`extract_phase_from_cp` and `controlled_phase_diag` exist per tensor); stacked views by gate kind, which `manifolds.group_by_manifold` and `stack_tensors` already provide for the optimiser.
- **Freezing by gate kind.** `frozen_indices` for "every one-qubit gate", "every gate on the row register", and so on, from the program. The completion models become: phase-only (a parameter view), all four phases (`QFTBasis` with the Hadamards frozen), and rotations (`QFTBasis` unfrozen).
- **The pixel-frame adapter.** In Yao's qubit order `QFTBasis` at initialisation is the DFT of the bit-reversed image. With `Pi` the bit reversal of both axes, `forward_transform(X) == U^T (Pi X Pi) U` for the standard-frame operator `U`. A small adapter exposes the standard frame, because the paper's seeded masks are drawn in it. `QFTBasis` itself does not change.
- **Operator-level coherence** from #35: `operator_coherence`, `flat_modulus_deviation`, the sampled check, and `is_flat_modulus` without the hidden relative tolerance.

## 5. Migration plan

Each step is its own pull request. Gate for every step: the full suite, the parity tests, the 90% coverage gate, the docs build, and the characterisation tests from step 0.

| Step | Change | Numbers move? |
|---|---|---|
| 0 | **Characterisation tests only.** Pin what must not change: initial tensors and their order for every basis (bit-exact), forward and inverse outputs on fixed inputs, pytree round trips, dataclass equality and `repr`, `num_parameters`, JSON bytes and hash, the indices `freeze_as_blocked` returns, a short training trajectory per optimiser path. Snapshots are generated from `main` before anything else changes. | no |
| 1 | **`Program` and `apply_program`**, with today's arithmetic. `compile_circuit` becomes a thin wrapper; instances share compiled code. | no, bit-exact |
| 2 | **Applier capabilities:** batch axes, image dtype, explicit one-qubit arithmetic, and the `U4` decision from Q1. | yes, at rounding level; the only step that does |
| 3 | **Emitters:** one QFT skeleton, one two-qubit factory. | no, tensors bit-exact |
| 4 | **`CircuitBasis`:** the seven classes collapse; `freeze_as_blocked`, `viz` and `coherence` read the program. | no |
| 5 | **Leftovers:** `loss._apply_circuit`, the einsum builder and cache to the tests, `CLAUDE.md`. | no |
| 6 | **Hooks:** parameter views, freezing by kind, the frame adapter, operator-level coherence. | additions only |
| then | **Completion stack re-cut** on the new core: solver, training, memory-bounded solver, evaluation, the remaining families, baselines, docs. | new code |

Step 2 is deliberately separate from step 1: the structural change is verified bit-exact before any arithmetic changes, so a regression in either is attributable.

## 6. What the completion stack becomes

| Old module | After the refactor |
|---|---|
| `transform` (kernel, `theta0`, `gate_pairs`) | `apply_program` on the QFT program plus the frame adapter; the angles are a parameter view |
| `bridge` | gone: there is one representation |
| `coherence` | the core module, already merged in #35's form |
| `families/phases`, `general`, `shared` | parameter views and freeze masks over `QFTBasis`, tens of lines each |
| `families/riemannian` `skew` / `cayley` | to be checked against `manifolds.UnitaryManifold.project` / `retract` before keeping a second copy |
| `adam` | stays: plain Adam is a different algorithm from Riemannian Adam. It moves to `optimizers/`, with the gradient conjugation fix |
| `training.adam_loop`, `minibatches` | `training/`; overlap with `train_basis_batched` to be resolved there |
| `solver.hard_k`, `comp_loss` | one thresholding function shared with `loss.topk_truncate`; `comp_loss` is `MSELoss(k)` |
| `separable`, `apply_dense` | stay with the families that are not circuits (dense unitary, butterfly, transform learning) |
| `solver` (IHT), `unroll`, `metrics`, `protocol`, `data`, `butterfly`, `transform_learning`, `baselines` | unchanged in content |

## 7. Risks and open questions

**Q1. Arithmetic for dense two-qubit gates.** The 16-term explicit form emits about 40 times more operations. With the package's disk cache warm that costs about 2 s at first call instead of 0.3 s, mostly tracing; cold it costs 15 s once per machine and graph, and on every CI run (section 3.4). The cost caching cannot remove is the CPU steady state at small sizes, 1.5 to 3.5x slower (section 3.3). Options: keep `tensordot` for `U4` and require the exact-matmul context for float32; or apply the gate on a merged axis of length 4, which needs far fewer slice and stack operations. Decide in step 2 from measurements of both.

**Q2. One-qubit arithmetic on the CPU at small sizes.** About 1.4x slower per gradient at 64x64, against 5 to 10x faster on the GPU at 512x512. Compilation doubles when cold and is unchanged when warm. The package's Julia-parity work runs small and on the CPU. Before committing, measure end-to-end wall-clock of the parity examples and of one `pdft-benchmarks` configuration. If it matters, the arithmetic can be selected per call rather than fixed.

**Q3. Rounding-level changes.** Step 2 moves results at 1e-16. All 280 tests pass on the CPU under it; the GPU and the benchmarks repository's recorded numbers have not been checked.

**Q4. Frame and reproducibility of the paper's numbers.** The adapter keeps the standard frame, so the paper's masks and protocol are reproduced. Bit-identity with the paper repository's trajectories, which the port currently has for several families, would be lost, because the operation order differs. Agreement would be statistical. Note that the port is already not seed-reproducible from the DFT starting point for even budgets `k`, where conjugate-paired coefficients tie.

**Q5. Pytree structure equality.** Today the closures `code` and `inv_code` sit in the pytree aux data and compare by identity, so two instances of the same basis have unequal tree structures. With a hashable `Program` there instead, they compare equal. This is an improvement (fewer retraces) but it is a change in observable behaviour.

**Q6. Public surface.** Removing `build_circuit_einsum` and `optimize_code_cached` from `pdft.circuit` is an API change, handled by a deprecation alias. Everything re-exported from `pdft` keeps its name and signature.

**Out of scope.** The two Riemannian Adam implementations (`optimizers/adam.py` and `training/adam_step.py`, duplicated on purpose for XLA), the JSON schema, and the upstream pin.

## 8. Decisions requested

1. Is the direction accepted, in this step order?
2. Step 2: is a rounding-level change to every basis's numbers acceptable, given that all parity tests pass?
3. Q1: which route for dense two-qubit gates, or defer to the step 2 measurements?
4. The legacy einsum builder: move to the tests as a reference oracle (recommended), or delete?
5. Q4: is bit-identity with the paper repository required, or is statistical agreement enough?
6. #35: close it and carry its coherence additions and the `apply_dense` fix into steps 6 and the re-cut stack, or keep it open as a reference? #34 and #36 to #44 stay as the reference tree for the re-cut either way; #45 is superseded.

## Appendix: how the numbers were produced

The one-qubit change, as installed for measurement in place of the `H` branch of `_stepped_apply`:

```python
U = T.T if inverse else T
a, b = jnp.take(pic, 0, axis=ax), jnp.take(pic, 1, axis=ax)
pic = jnp.stack([U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b], axis=ax)
```

The two-qubit change takes the four slices of the two wires and forms `out[oc, ot] = sum M[oc, ot, ic, it] * s[ic][it]`, with `M = T` forward and `M = transpose(T, (2, 3, 0, 1))` inverse. `CRY` applies the one-qubit form to the control-one half.

- Tests: the full suite run in one process after installing the patch, `JAX_PLATFORMS=cpu`.
- Timings: first call includes tracing and compilation; steady state is the mean of 20 to 200 repetitions after it, with `block_until_ready`.
- Solver gradients: `K` steps of `X <- where(obs, Y, Re S(H_k(A(X))))` under `lax.scan`, each step wrapped in `jax.checkpoint`, 10% of pixels observed, differentiated with respect to every tensor.
- Equivalence with the completion kernel: `QFTBasis.forward_transform(X)` against `conj(analysis(Pi X Pi))` at the DFT parameters.
