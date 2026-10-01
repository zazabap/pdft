"""Mutual coherence of a learned basis with the pixel basis.

For compression, the only thing that matters about a basis is how few
coefficients it needs. For any task that recovers an image from a *subset of
its pixels* --- inpainting, completion, compressed sensing --- a second
quantity governs the outcome, and it is not sparsity::

    mu(U) = N * max_ij |U_ij|^2   in [1, N]

with N the operator dimension. mu = 1 is maximal incoherence with the pixel
basis, the most favourable case for recovery from pointwise samples; mu = N is
the worst, an atom supported on a single pixel, invisible to any sample set
that misses it. Sample-complexity bounds for recovery from random pixel
observations scale linearly in mu (Candes & Plan 2011, Krahmer & Ward 2014).

A trained basis can drift in mu while its compression loss improves, so the
quantity is worth being able to measure. This module measures it --- and, more
usefully, certifies when it cannot drift at all.

The guarantee
-------------
Let the circuit act on n wires, and suppose every gate is diagonal in the
computational basis except for exactly one Hadamard per wire. Then::

    |U_ij| = N^{-1/2}   for every i, j and every parameter value,

so mu(U) = 1 identically over the whole parameter space, and sqrt(N) U is a
complex Hadamard matrix (Tadej & Zyczkowski 2006).

*Proof.* Track the amplitude vector produced by an input basis vector.
Diagonal gates multiply amplitudes by unit moduli and never move amplitude
between the two values of any wire; a permutation permutes. Consider the moment
the Hadamard on wire q is applied. Every earlier gate is diagonal or a Hadamard
on a different wire, so none has moved amplitude across wire q, and the vector
is supported on one value of it: one branch carries a, the other 0. Since
H (a, 0)^T = (a, a)^T / sqrt(2), the gate replaces each modulus by two equal
moduli ``|a|/sqrt(2)`` and never combines unequal ones. Each wire receives exactly
one Hadamard, so every amplitude ends at 2^{-n/2}, and the remaining diagonal
gates and permutations preserve moduli. QED

The practical consequence is a classification of this package's bases by which
gates are left free during training:

  * free only the controlled-phase gates --- the compact-CP tensors --- and
    mu is pinned at 1 no matter how hard the basis is trained, on any
    objective, with no penalty, constraint or monitoring;
  * free the Hadamard / U(4) gates as well and the guarantee is void. It is
    void in practice, not only in principle: randomising them on an (m, n) =
    (3, 3) QFTBasis reaches mu = 24.8 of a possible 64, and on RichBasis ---
    which has no diagonal gates at all --- mu = 32.5.

`certify_flat_modulus` performs that check for a given training configuration,
taking the same `frozen_indices` that `train_basis_batched` accepts, so the
question "does this training run preserve incoherence?" can be answered before
the run rather than measured after it. `sampled_flat_modulus` measures the same
thing instead of proving it, by drawing values for the tensors left trainable.

`operator_coherence` and `flat_modulus_deviation` are the two quantities for
an explicit matrix, kept inside JAX so a loss may carry them.

Discovered while applying this package's basis family to image *completion*,
where the reversal matters: the transform that compresses an image best is not
the one that completes it best.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .bases.core import program_of, with_tensors
from .circuit.builder import is_compact_cp
from .manifolds import PhaseManifold, classify_manifold

Array = jax.Array

__all__ = [
    "FlatModulusCertificate",
    "certify_flat_modulus",
    "coherence",
    "dense_operator",
    "diagonal_tensor_indices",
    "flat_modulus_deviation",
    "is_flat_modulus",
    "operator_coherence",
    "sampled_flat_modulus",
]


def dense_operator(basis) -> Array:
    """The basis as an explicit matrix acting on `vec(image)`.

    Column j is the transform of the image that is 1 at pixel j and 0
    elsewhere, so the result is (2^m 2^n) x (2^m 2^n).

    This is a diagnostic, not a code path: it costs N^2 transforms of an
    N-pixel image and is meant for small m, n. Nothing in training needs it ---
    `certify_flat_modulus` gives the parameter-space guarantee without forming
    it.
    """
    rows, cols = basis.image_size
    dim = rows * cols
    eye = jnp.eye(dim, dtype=jnp.complex128)
    return jax.vmap(lambda e: basis.forward_transform(e.reshape(rows, cols)).reshape(-1))(eye).T


def operator_coherence(u: Array) -> Array:
    """``mu`` of an explicit matrix, as a traceable scalar.

    `coherence` takes a basis and returns a Python float; this is the same
    quantity kept inside JAX, so a loss may carry ``lam * mu`` under `jax.jit`.
    """
    return u.shape[0] * jnp.max(jnp.abs(u) ** 2)


def flat_modulus_deviation(u: Array) -> Array:
    """``max_ij | |U_ij| - N^{-1/2} |``, the residual of the guarantee.

    Worth reporting wherever a number is quoted: it says how exactly the
    guarantee holds, not merely that it does. Traceable.
    """
    return jnp.max(jnp.abs(jnp.abs(u) - u.shape[0] ** -0.5))


def coherence(basis, operator: Array | None = None) -> float:
    """``mu(U) = N max_ij |U_ij|^2``, in [1, N]. 1 is maximal incoherence.

    Pass `operator` to reuse a matrix from `dense_operator`.
    """
    u = dense_operator(basis) if operator is None else operator
    return float(operator_coherence(u))


def is_flat_modulus(basis, operator: Array | None = None, atol: float = 1e-8) -> bool:
    """True if ``|U_ij| = N^{-1/2}`` everywhere, i.e. sqrt(N) U is complex Hadamard.

    The comparison is `jnp.allclose` with this `atol`, so its default
    relative slack of ``1e-5`` applies on top: a basis held in single
    precision, which sits about ``2e-8`` from flat, passes. For the number
    itself use `flat_modulus_deviation`.
    """
    u = dense_operator(basis) if operator is None else operator
    return bool(jnp.allclose(jnp.abs(u), u.shape[0] ** -0.5, atol=atol))


def diagonal_tensor_indices(basis) -> list[int]:
    """Indices of the tensors that are diagonal in the computational basis.

    These are the compact controlled-phase gates, `[[1, 1], [1, e^{i phi}]]`,
    whose only freedom is a phase. Freeing exactly these is what preserves
    mu == 1; see the module docstring.

    Read from the basis's program, not guessed from tensor values: a gate of
    kind ``"CP"`` is applied as a diagonal whatever its entries have become.
    A basis that is not one of the package's circuits has no program; for it
    the tensors' values decide, as they did for every basis before.
    """
    try:
        program = program_of(basis)
    except AttributeError:
        return [i for i, t in enumerate(basis.tensors) if is_compact_cp(t)]
    return program.tensor_indices(kind="CP")


@dataclass(frozen=True)
class FlatModulusCertificate:
    """Whether mu == 1 holds over the whole reachable parameter set.

    `holds` is the answer; `reason` says why in one line. `mu` is the coherence
    at the basis's current parameters, and `offending_indices` names the
    trainable tensors that are not diagonal, which are exactly the ones that
    would have to be frozen for the guarantee to apply.
    """

    holds: bool
    reason: str
    mu: float
    offending_indices: list[int] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.holds


def certify_flat_modulus(
    basis,
    frozen_indices: list[int] | None = None,
    *,
    atol: float = 1e-8,
) -> FlatModulusCertificate:
    """Certify that training cannot raise mu above 1.

    Two conditions, both necessary. The basis must be flat-modulus *now*, and
    every tensor left trainable must be diagonal --- then by the proposition in
    the module docstring, mu == 1 for every value those parameters can take.

    `frozen_indices` is the same argument `train_basis_batched` accepts, so::

        cert = certify_flat_modulus(basis, frozen_indices=frozen)
        if not cert:
            raise ValueError(cert.reason)
        result = train_basis_batched(basis, frozen_indices=frozen, ...)

    Frozen non-diagonal gates are fine: a fixed Hadamard is what the
    proposition assumes. It is *training* them that voids it.
    """
    operator = dense_operator(basis)
    mu = coherence(basis, operator)

    if not is_flat_modulus(basis, operator, atol=atol):
        return FlatModulusCertificate(
            holds=False,
            reason=(
                f"basis is not flat-modulus at its current parameters "
                f"(mu = {mu:.6g}); the guarantee applies to circuits with one "
                f"Hadamard per wire and diagonal gates elsewhere"
            ),
            mu=mu,
        )

    frozen = set(frozen_indices or [])
    diagonal = set(diagonal_tensor_indices(basis))
    offending = [i for i in range(len(basis.tensors)) if i not in frozen and i not in diagonal]

    if offending:
        return FlatModulusCertificate(
            holds=False,
            reason=(
                f"mu = {mu:.6g} now, but {len(offending)} trainable tensor(s) "
                f"are not diagonal, so training may raise it; freeze indices "
                f"{offending} to obtain the guarantee"
            ),
            mu=mu,
            offending_indices=offending,
        )

    return FlatModulusCertificate(
        holds=True,
        reason=(
            f"mu == 1 for every reachable parameter value: all "
            f"{len(basis.tensors) - len(frozen)} trainable tensor(s) are diagonal"
        ),
        mu=mu,
    )


def _random_point(tensor: Array, rng: np.random.Generator) -> Array:
    """A random point of the manifold `tensor` is trained on, with its shape and dtype.

    The manifold is the one `classify_manifold` assigns, as in the trainers:
    unit-modulus entries for a diagonal gate, a Haar unitary for the others.
    """
    if isinstance(classify_manifold(tensor), PhaseManifold):
        drawn = np.exp(1j * rng.uniform(-np.pi, np.pi, tensor.shape))
    else:
        d = round(float(np.sqrt(tensor.size)))
        q, r = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
        drawn = (q * (np.diag(r) / np.abs(np.diag(r)))).reshape(tensor.shape)
    return jnp.asarray(drawn, dtype=tensor.dtype)


def sampled_flat_modulus(
    basis,
    frozen_indices: list[int] | None = None,
    *,
    trials: int = 8,
    atol: float | None = None,
    seed: int = 0,
) -> dict:
    """Measure the guarantee over drawn values of the tensors left trainable.

    Each trial replaces every tensor not in `frozen_indices` with a random
    point of its manifold and measures the dense operator. The check holds
    when every draw is flat-modulus, which is what distinguishes a basis that
    happens to be incoherent from a training configuration that cannot leave
    the complex Hadamard set. `certify_flat_modulus` proves the same thing
    from the circuit's structure; this measures it, and says how far a
    configuration without the guarantee drifts.

    Returns the verdict, the worst deviation and the worst mu seen. `atol`
    bounds the worst deviation; left out, it is ``1e-10``, or ``1e-6`` when a
    tensor is held in single precision (whose draws sit about ``3e-8`` from
    flat).
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if atol is None:
        single = any(t.dtype == jnp.complex64 for t in basis.tensors)
        atol = 1e-6 if single else 1e-10
    rng = np.random.default_rng(seed)
    frozen = set(frozen_indices or [])
    worst_deviation, worst_mu = 0.0, 0.0
    for _ in range(trials):
        drawn = [t if i in frozen else _random_point(t, rng) for i, t in enumerate(basis.tensors)]
        u = dense_operator(with_tensors(basis, drawn))
        worst_deviation = max(worst_deviation, float(flat_modulus_deviation(u)))
        worst_mu = max(worst_mu, float(operator_coherence(u)))
    return {
        "holds": worst_deviation <= atol,
        "worst_deviation": worst_deviation,
        "worst_mu": worst_mu,
        "trials": trials,
    }
