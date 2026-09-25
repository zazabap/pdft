"""The unrestricted isometric tensor network of arXiv:2608.00053.

That work generalises the QFT circuit by relaxing every gate within its own
manifold: "Each fixed Hadamard becomes an arbitrary unitary, and each
conditional phase keeps its diagonal form while freeing its diagonal entries."
Riemannian optimization then "guarantees that every optimization step returns
each gate to its manifold, so the tensor network remains isometric throughout
training." (That is exactly what :class:`pdft.QFTBasis` trains under
:func:`pdft.train_basis`; :mod:`pdft.completion.bridge` converts between the
two representations.)

This module implements that family so it can be run on completion and compared
against the phase-only restriction:

    phase-only (A)      fixed Hadamards, 1 free phase per CP gate
                        n(n-1)/2 params/axis, unconstrained, mu == 1 always
    diagonals (B)       fixed Hadamards, 4 free phases per two-qubit gate
                        2n(n-1) params/axis, unconstrained, mu == 1 always
    rotations (C)       free U(2) per bit, 4 free phases per two-qubit gate
                        4n + 2n(n-1) params/axis, Riemannian, mu free to drift

All reduce to the DFT at initialisation, so the comparison is nested.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..transform import complex_dtype, gate_pairs, n_params, theta0

_H2 = jnp.asarray(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0), dtype=jnp.complex128)


def init_general(n: int) -> dict:
    """Initialise at the DFT: Hadamards, and CP phases (0, 0, 0, theta0)."""
    th = np.asarray(theta0(n))
    phi = np.zeros((n_params(n), 4))
    phi[:, 3] = th  # diag(1, 1, 1, e^{i theta}) = a CP gate
    return {"g": jnp.broadcast_to(_H2, (n, 2, 2)).copy(), "phi": jnp.asarray(phi)}


@functools.partial(jax.jit, static_argnames=("n", "adjoint", "axis"))
def apply_general(x, params, n: int, adjoint: bool = False, axis: int = -1):
    """Apply the generalised circuit along one axis of length 2**n."""
    cdtype = complex_dtype(x)
    g = params["g"].astype(cdtype)
    phi = params["phi"]
    x = jnp.moveaxis(x.astype(cdtype), axis, -1)
    lead = x.shape[:-1]
    nl = len(lead)
    t = x.reshape(lead + (2,) * n)

    def qax(q):
        return nl + q

    def one_qubit(t, q, U):
        a = jnp.take(t, 0, axis=qax(q))
        b = jnp.take(t, 1, axis=qax(q))
        return jnp.stack([U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b], axis=qax(q))

    def two_qubit_diag(t, p, q, ph):
        # One broadcast multiply rather than four scatters: with n(n-1)/2 gates
        # per axis, four applications per solver step and K steps under a
        # checkpointed scan, the scatter version produces an XLA graph that
        # takes longer to compile than the experiment takes to run.
        ph4 = jnp.exp(1j * ph.astype(cdtype)).reshape(2, 2)  # [b_p, b_q]
        shape = [1] * t.ndim
        shape[qax(q)] = 2
        shape[qax(p)] = 2  # q < p, so b_q precedes b_p
        return t * ph4.T.reshape(shape)

    def bitreverse(t):
        return jnp.transpose(t, tuple(range(nl)) + tuple(nl + n - 1 - i for i in range(n)))

    pairs = gate_pairs(n)
    if not adjoint:
        k = 0
        for q in range(n):
            t = one_qubit(t, q, g[q])
            for p in range(q + 1, n):
                t = two_qubit_diag(t, p, q, phi[k])
                k += 1
        t = bitreverse(t)
    else:
        t = bitreverse(t)
        k = len(pairs)
        for q in reversed(range(n)):
            for p in reversed(range(q + 1, n)):
                k -= 1
                t = two_qubit_diag(t, p, q, -phi[k])
            t = one_qubit(t, q, jnp.conj(g[q]).T)
    return jnp.moveaxis(t.reshape(lead + (2**n,)), -1, axis)


def analysis_g(X, pr, pc, n):
    C = apply_general(X, pr, n, adjoint=True, axis=-2)
    return apply_general(C, pc, n, adjoint=True, axis=-1)


def synthesis_g(C, pr, pc, n):
    X = apply_general(C, pr, n, adjoint=False, axis=-2)
    return apply_general(X, pc, n, adjoint=False, axis=-1)


# --------------------------------------------------------------------------
# rectangular images: the two registers need not have the same width
#
# apply_general only reshapes the axis it is given, so nothing had to change to
# support 2^nr x 2^nc --- which matters because a 4096 x 512 strip carries a
# complete n = 12 register at an eighth of the memory of a 4096^2 image, and
# training memory is what bounds the resolution the basis can be fitted at.


def analysis_rect(X, pr, pc, nr, nc):
    C = apply_general(X, pr, nr, adjoint=True, axis=-2)
    return apply_general(C, pc, nc, adjoint=True, axis=-1)


def synthesis_rect(C, pr, pc, nr, nc):
    X = apply_general(C, pr, nr, adjoint=False, axis=-2)
    return apply_general(X, pc, nc, adjoint=False, axis=-1)


def reconstruct_rect(pr, pc, Y, obs, nr, nc, k, K, mode="hard", remat=True):
    """The unrolled solver on a 2^nr x 2^nc image."""
    from ..solver import iht

    return iht(
        lambda X: analysis_rect(X, pr, pc, nr, nc),
        lambda C: synthesis_rect(C, pr, pc, nr, nc),
        Y,
        obs,
        k,
        K,
        mode,
        remat,
    )


@functools.partial(jax.jit, static_argnames=("n", "K", "mode", "remat"))
def reconstruct_g(pr, pc, Y, obs, n, k, K, mode="hard", remat=True):
    """The square case of reconstruct_rect, jitted --- the evaluation solver.
    k is traced (solver.kth_largest), so a new budget does not recompile."""
    return reconstruct_rect(pr, pc, Y, obs, n, n, k, K, mode, remat)


def unitary_general(params, n):
    """Form the relaxed circuit's matrix explicitly. Diagnostics only."""
    from ..coherence import dense_operator

    return dense_operator(lambda e: apply_general(e, params, n, adjoint=False, axis=0), n)


def coherence_general(params, n):
    """mu of the relaxed circuit. See :mod:`pdft.completion.coherence`."""
    from ..coherence import coherence as _mu

    return _mu(unitary_general(params, n))


def evaluate_general(par, images, n, p, frac, K, seed, mode="hard"):
    """Held-out PSNR of a {"g", "phi"} basis under the shared protocol."""
    from ..protocol import evaluate

    return evaluate(
        lambda Y, obs, k: reconstruct_g(
            par["r"], par["c"], Y, obs, n, k, K, mode=mode, remat=False
        ),
        images,
        p,
        frac,
        seed,
    )


def count_params(n: int) -> int:
    """Real DOF: U(2) per bit (4 each) + 4 phases per two-qubit gate."""
    return 4 * n + 4 * n_params(n)


def count_params_b(n: int) -> int:
    """Real DOF of Model B: four free diagonal entries per two-qubit gate.

    The 4n reals of the one-qubit gates are *not* counted, because they are not
    free --- freeing them is Model C, which loses Proposition 1 and needs a
    retraction. count_params counts C, so it is deliberately not reused here.
    """
    return 4 * n_params(n)


def train_general(
    images,
    n,
    k,
    K=100,
    p=0.10,
    steps=200,
    lr=2e-3,
    mode="hard",
    model="B",
    batch=2,
    seed=0,
    log_every=25,
    remat=True,
    verbose=True,
):
    """Adam on the diagonal phases, with the one-qubit gates frozen at the
    Hadamards --- Model B, the paper's headline family.

    g is closed over rather than handed to the optimiser, which is what keeps
    this Model B and not Model C. model="A" ties the four diagonal entries of
    each gate to a single controlled phase by masking the other three out of
    the gradient, so both restrictions ride the same circuit and step. Freeing
    g as well is Model C --- which loses Proposition 1 and needs a retraction;
    see train_c below.

    The batch/mask schedule is adam_loop's, shared with every other trained
    family. Returns (params, history) with params in the {"g", "phi"} form the
    rest of this module consumes.
    """
    from ..training import adam_loop

    if model not in ("A", "B"):
        raise ValueError(f"model must be 'A' or 'B', got {model!r}; Model C is train_c")
    images = jnp.asarray(images)
    init = init_general(n)
    gs = init["g"]
    phis = {"r": init["phi"], "c": init["phi"]}
    grad_mask = None
    if model == "A":
        m = np.zeros((n_params(n), 4))
        m[:, 3] = 1.0  # only the controlled phase moves
        grad_mask = {"r": jnp.asarray(m), "c": jnp.asarray(m)}

    def loss_fn(phis, X, obs):
        pr = {"g": gs, "phi": phis["r"]}
        pc = {"g": gs, "phi": phis["c"]}

        def f(y, o):
            return reconstruct_rect(pr, pc, y, o, n, n, k, K, mode, remat)

        Xh = jax.vmap(f)(X * obs, obs)
        return jnp.mean((Xh - X) ** 2)

    def monitor(phis):
        return {
            "mu_r": float(coherence_general({"g": gs, "phi": phis["r"]}, n)),
            "mu_c": float(coherence_general({"g": gs, "phi": phis["c"]}, n)),
        }

    phis, history = adam_loop(
        images,
        phis,
        loss_fn,
        lr=lr,
        steps=steps,
        p=p,
        batch=batch,
        seed=seed,
        grad_mask=grad_mask,
        monitor=monitor,
        log_every=log_every,
        verbose=verbose,
    )
    return {a: {"g": gs, "phi": phis[a]} for a in ("r", "c")}, history


def cayley_u2(G, A, tau):
    """Cayley retraction on U(2): G <- G (I + tau/2 A)^{-1} (I - tau/2 A).

    Exactly unitary for skew-Hermitian A, and G (I - tau A) to first order,
    so with A the Riemannian gradient's generator (see train_c) tau > 0 is
    the descent direction.
    """
    eye = jnp.eye(2, dtype=G.dtype)
    return G @ jnp.linalg.solve(eye + 0.5 * tau * A, eye - 0.5 * tau * A)


def riemannian_generator(G, E):
    """A = G^H E - E^H G for a batch of U(2) gates and Euclidean gradients E,
    symmetrised so it is exactly skew-Hermitian."""
    A = jnp.einsum("qij,qik->qjk", jnp.conj(G), E) - jnp.einsum("qij,qik->qjk", jnp.conj(E), G)
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, 1, 2)))


def train_c(
    images,
    n,
    k,
    K,
    p,
    steps,
    lr_phi,
    lr_g,
    seed,
    batch=2,
    log_every=50,
    verbose=True,
    remat=True,
):
    """Model C: Adam on the phases, Cayley SGD keeping each U(2) on its
    manifold --- the prior work's training, on the prior work's family.

    The one-qubit gates G live on U(2), whose tangent space at G is
    {G Om : Om skew-Hermitian}. With E the Euclidean gradient of the loss in
    G (jax.grad on a real loss of a complex input returns its conjugate, which
    the conj below undoes), the Riemannian gradient is G A / 2 with
    A = G^H E - E^H G, and the descent step is cayley_u2(G, A, +lr_g).

    A check from exactly theta0 does not expose the sign of the step, because
    the first-order term vanishes there and a top-k tie-break jump masks it;
    the tests verify descent from a perturbed point.

    Returns params in the {"g", "phi"} form the rest of this module consumes;
    the batch/mask draw order is the same numpy schedule as adam_loop's, so
    C's minibatches match A's and B's at the same seed.
    """
    from ..adam import adam_init, adam_update, apply_updates

    images = jnp.asarray(images)  # the caller's dtype sets the precision
    par = {a: init_general(n) for a in ("r", "c")}
    st = adam_init({a: par[a]["phi"] for a in ("r", "c")})

    def loss(phis, gs, X, obs):
        pr = {"g": gs["r"], "phi": phis["r"]}
        pc = {"g": gs["c"], "phi": phis["c"]}

        def f(y, o):
            return reconstruct_g(pr, pc, y, o, n, k, K, remat=remat)

        Xh = jax.vmap(f)(X * obs, obs)
        return jnp.mean((Xh - X) ** 2)

    @jax.jit
    def step_fn(par, st, X, obs):
        phis = {a: par[a]["phi"] for a in ("r", "c")}
        gs = {a: par[a]["g"] for a in ("r", "c")}
        v, (gphi, gg) = jax.value_and_grad(loss, argnums=(0, 1))(phis, gs, X, obs)
        upd, st = adam_update(gphi, st, lr_phi)
        phis = apply_updates(phis, upd)
        newg = {}
        for a in ("r", "c"):
            G, E = gs[a], jnp.conj(gg[a])
            A = riemannian_generator(G, E)
            newg[a] = jax.vmap(lambda g, aa: cayley_u2(g, aa, lr_g))(G, A)
        return {a: {"g": newg[a], "phi": phis[a]} for a in ("r", "c")}, st, v

    rng = np.random.default_rng(seed)
    for it in range(steps):
        idx = rng.choice(len(images), size=min(batch, len(images)), replace=False)
        X = images[idx]
        obs = jnp.asarray(rng.random(X.shape) < p)
        par, st, v = step_fn(par, st, X, obs)
        if verbose and (it % log_every == 0 or it == steps - 1):
            print(
                f"    {it:4d}  loss {float(v):.6e}  mu {float(coherence_general(par['r'], n)):.4f}",
                flush=True,
            )
    return par
