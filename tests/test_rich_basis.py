"""Tests for ``pdft.RichBasis`` — H + learnable U(4) gate-set basis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pdft
from pdft.manifolds import Unitary2qManifold, UnitaryManifold, group_by_manifold


def _rand_pic(m: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n))).astype(
        np.complex128
    )


# ---- construction -----------------------------------------------------------


def test_rich_basis_rejects_zero_dim():
    with pytest.raises(ValueError, match=">= 1"):
        pdft.RichBasis(m=0, n=1)


def test_rich_basis_image_size_and_param_count():
    b = pdft.RichBasis(m=3, n=3)
    assert b.image_size == (8, 8)
    # 3 H per dim (2x2 each = 4 elements) + 3 U(4) per dim (16 elements each)
    # total raw elements: 2 * (3*4 + 3*16) = 120
    # FREE real params (manifold-aware): 2 * (3 * 3 + 3 * 15) = 108 (= 54 per dim)
    # Below SU(8) dim of 63 — strict submanifold.
    assert b.num_parameters == 2 * (3 * 4 + 3 * 16)


def test_rich_basis_round_trip():
    b = pdft.RichBasis(m=3, n=3)
    pic = jnp.asarray(_rand_pic(3, 3))
    fwd = b.forward_transform(pic)
    rec = b.inverse_transform(fwd)
    assert jnp.allclose(rec, pic, atol=1e-10)


# ---- equivalence to QFT at init --------------------------------------------


def test_rich_basis_init_matches_qft_forward():
    """At init, RichBasis must produce IDENTICAL forward outputs to QFTBasis.

    This is the key property: RichBasis embeds QFT as a special case (each
    U(4) gate starts at the diagonal CP value). So the optimiser starts
    exactly where plain QFT starts and any improvement is real.
    """
    rich = pdft.RichBasis(m=3, n=3)
    qft = pdft.QFTBasis(m=3, n=3)
    pic = jnp.asarray(_rand_pic(3, 3, seed=7))
    rich_out = rich.forward_transform(pic)
    qft_out = qft.forward_transform(pic)
    assert jnp.allclose(rich_out, qft_out, atol=1e-12)


def test_rich_basis_init_matches_qft_inverse():
    rich = pdft.RichBasis(m=3, n=3)
    qft = pdft.QFTBasis(m=3, n=3)
    pic = jnp.asarray(_rand_pic(3, 3, seed=11))
    rich_inv = rich.inverse_transform(pic)
    qft_inv = qft.inverse_transform(pic)
    assert jnp.allclose(rich_inv, qft_inv, atol=1e-12)


# ---- manifold classification -----------------------------------------------


def test_rich_basis_tensors_classify_correctly():
    b = pdft.RichBasis(m=3, n=3)
    groups = group_by_manifold(b.tensors)
    # Expect: 6 H gates → UnitaryManifold(d=2); 6 U(4) gates → Unitary2qManifold
    types = sorted(type(k).__name__ for k in groups)
    assert types == ["Unitary2qManifold", "UnitaryManifold"]
    # H gates are (2, 2); U(4) tensors are (2, 2, 2, 2).
    u_group = next(idxs for k, idxs in groups.items() if isinstance(k, UnitaryManifold))
    u4_group = next(idxs for k, idxs in groups.items() if isinstance(k, Unitary2qManifold))
    assert all(b.tensors[i].shape == (2, 2) for i in u_group)
    assert all(b.tensors[i].shape == (2, 2, 2, 2) for i in u4_group)
    assert len(u_group) == 6  # 3 H per dim
    assert len(u4_group) == 6  # 3 U(4) per dim


def test_unitary2q_manifold_project_preserves_tangent_skew_on_4x4():
    """Project sends gradients to the Lie algebra; skew U^H * proj(G) on the
    4x4 reshape."""
    rng = np.random.default_rng(31)
    # 2 random U(4) gates stacked
    U_4x4 = []
    for _ in range(2):
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        U_4x4.append(Q)
    T = jnp.stack([jnp.asarray(u.reshape(2, 2, 2, 2)) for u in U_4x4], axis=-1)
    G = jnp.asarray(
        rng.standard_normal((2, 2, 2, 2, 2)) + 1j * rng.standard_normal((2, 2, 2, 2, 2))
    )
    P = Unitary2qManifold().project(T, G)
    assert P.shape == T.shape


def test_unitary2q_manifold_retract_preserves_unitarity():
    """Retract from a unitary point along a tangent should remain on U(4)."""
    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    T = jnp.asarray(Q).reshape(2, 2, 2, 2)[..., None]  # (2,2,2,2,1)
    Xi_mat = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) * 0.1
    Xi = jnp.asarray(Xi_mat).reshape(2, 2, 2, 2)[..., None]
    new_T = Unitary2qManifold().retract(T, Xi, 0.1)
    new_4x4 = np.asarray(new_T[..., 0]).reshape(4, 4)
    err = np.max(np.abs(new_4x4 @ new_4x4.conj().T - np.eye(4)))
    assert err < 1e-8


# ---- gradients & training --------------------------------------------------


def test_rich_basis_grad_finite():
    b = pdft.RichBasis(m=2, n=2)
    pic = jnp.asarray(_rand_pic(2, 2, seed=44))

    def loss_fn(tensors):
        from pdft.loss import _apply_circuit

        out = _apply_circuit(tensors, b.code, b.m, b.n, pic)
        return jnp.sum(jnp.abs(out) ** 2)

    grads = jax.grad(loss_fn)(list(b.tensors))
    for g in grads:
        assert jnp.all(jnp.isfinite(g))


def test_rich_basis_trains_via_train_basis_batched():
    """train_basis_batched runs end-to-end on RichBasis and reduces loss."""
    b = pdft.RichBasis(m=2, n=2)
    rng = np.random.default_rng(101)
    imgs = [
        (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(np.complex128)
        for _ in range(8)
    ]
    res = pdft.train_basis_batched(
        b,
        dataset=imgs,
        loss=pdft.MSELoss(k=4),
        epochs=3,
        batch_size=2,
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        shuffle=False,
        seed=42,
    )
    losses = list(res.loss_history)
    assert len(losses) > 0
    head = losses[0]
    tail_mean = float(np.mean(losses[-max(1, len(losses) // 4) :]))
    assert tail_mean < head * 2.0


def test_rich_basis_block_wrapped_pytree_round_trip():
    """BlockedBasis(RichBasis(3,3), 5, 5) flattens + unflattens via JAX pytree."""
    inner = pdft.RichBasis(m=3, n=3)
    blocked = pdft.BlockedBasis(inner=inner, block_log_m=5, block_log_n=5)
    leaves, treedef = jax.tree_util.tree_flatten(blocked)
    blocked2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(blocked2.inner, pdft.RichBasis)
    pic = jnp.asarray(_rand_pic(8, 8, seed=22))
    assert jnp.allclose(blocked.forward_transform(pic), blocked2.forward_transform(pic), atol=1e-12)


def test_rich_basis_block_wrapped_trains():
    """The actual benchmark configuration: BlockedBasis(RichBasis(2,2), 1, 1)
    on a small 4x4 grid trains end-to-end without errors."""
    inner = pdft.RichBasis(m=2, n=2)
    basis = pdft.BlockedBasis(inner=inner, block_log_m=1, block_log_n=1)
    rng = np.random.default_rng(202)
    imgs = [
        (
            rng.standard_normal((2**basis.m, 2**basis.n))
            + 1j * rng.standard_normal((2**basis.m, 2**basis.n))
        ).astype(np.complex128)
        for _ in range(4)
    ]
    res = pdft.train_basis_batched(
        basis,
        dataset=imgs,
        loss=pdft.MSELoss(k=8),
        epochs=2,
        batch_size=2,
        optimizer="adam",
        validation_split=0.0,
        early_stopping_patience=10,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        shuffle=False,
        seed=42,
    )
    assert len(res.loss_history) > 0
