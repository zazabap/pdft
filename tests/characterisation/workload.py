"""A workload over pdft's public API, every output of which is recorded.

`compare_refs.py` runs it against the source tree of a git ref and against the
working tree and compares the two records bit for bit. It is the wider net
next to the snapshots: twenty basis configurations, both trainers in about 130
configurations, freezing, JSON, hashes, compression, coherence, `fit_to_dct`,
single-precision tensors, images of other shapes, keyword calls and error
messages. A snapshot pins a value forever; this compares two trees today, so
it can cover what would be too large or too machine-specific to commit.

Only names that exist on `main` at 102f5b6 are used, so the same script runs on
both sides of the refactor. Images are complex: real images at the symmetric
initial tensors give exactly tied coefficient pairs, and a tie is broken by
rounding.

    PYTHONPATH=<tree>/src python tests/characterisation/workload.py out.npz
"""

import contextlib
import io
import itertools
import json
import re
import sys
import zlib

import jax
import jax.numpy as jnp
import numpy as np

import pdft
from pdft.bases.circuit.rich import fit_to_dct
from pdft.coherence import certify_flat_modulus, coherence, dense_operator, diagonal_tensor_indices
from pdft.io import basis_hash, basis_to_dict, compress, compress_with_k, compressed_to_dict
from pdft.loss import loss_function


def collect() -> dict[str, np.ndarray]:
    out = {}

    def put(key, value):
        assert key not in out, key
        out[key] = np.asarray(value)

    def rng_for(name):
        return np.random.default_rng(zlib.crc32(name.encode()))

    def cimage(rng, shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    BASES = {
        "qft_2x2": lambda: pdft.QFTBasis(m=2, n=2),
        "qft_3x2": lambda: pdft.QFTBasis(m=3, n=2),
        "qft_1x3": lambda: pdft.QFTBasis(m=1, n=3),
        "entangled_3x2": lambda: pdft.EntangledQFTBasis(m=3, n=2, seed=1),
        "entangled_front_2x3": lambda: pdft.EntangledQFTBasis(
            m=2, n=3, seed=2, entangle_position="front"
        ),
        "entangled_phases_2x2": lambda: pdft.EntangledQFTBasis(
            m=2, n=2, entangle_phases=[0.3, -1.1]
        ),
        "tebd_cp_3x2": lambda: pdft.TEBDBasis(m=3, n=2, seed=1),
        "tebd_u4_2x2": lambda: pdft.TEBDBasis(m=2, n=2, seed=1, parametrization="u4"),
        "tebd_phases_2x3": lambda: pdft.TEBDBasis(m=2, n=3, phases=[0.1, 0.2, 0.3, 0.4, 0.5]),
        "mera_cp_4x2": lambda: pdft.MERABasis(m=4, n=2, seed=1),
        "mera_u4_2x2": lambda: pdft.MERABasis(m=2, n=2, seed=1, parametrization="u4"),
        "mera_1x2": lambda: pdft.MERABasis(m=1, n=2, seed=3),
        "dct4_o4_3x2": lambda: pdft.DCT4Basis(m=3, n=2),
        "dct4_controlled_3x2": lambda: pdft.DCT4Basis(m=3, n=2, parametrization="controlled"),
        "dct4_1x1": lambda: pdft.DCT4Basis(m=1, n=1),
        "rich_3x2": lambda: pdft.RichBasis(m=3, n=2),
        "real_rich_2x2": lambda: pdft.RealRichBasis(m=2, n=2),
        "blocked_qft": lambda: pdft.BlockedBasis(pdft.QFTBasis(m=2, n=2), 1, 1),
        "blocked_rich": lambda: pdft.BlockedBasis(pdft.RichBasis(m=1, n=2), 2, 1),
        "blocked_blocked": lambda: pdft.BlockedBasis(
            pdft.BlockedBasis(pdft.QFTBasis(m=1, n=1), 1, 1), 1, 1
        ),
    }

    def generic(basis, rng):
        return [jnp.asarray(cimage(rng, t.shape)) for t in basis.tensors]

    # 1. every basis: tensors, transforms, losses and their gradients
    for name, make in BASES.items():
        rng = rng_for(name)
        basis = make()
        m, n = basis.m, basis.n
        x = jnp.asarray(cimage(rng, basis.image_size))
        put(f"{name}/sizes", [m, n, len(basis.tensors), basis.num_parameters, *basis.image_size])
        for i, t in enumerate(basis.tensors):
            put(f"{name}/tensor{i}", t)
        put(f"{name}/forward", basis.forward_transform(x))
        put(f"{name}/inverse", basis.inverse_transform(x))
        put(f"{name}/forward_real", basis.forward_transform(jnp.real(x)))
        put(f"{name}/forward_f32", basis.forward_transform(jnp.real(x).astype(jnp.float32)))
        tensors = generic(basis, rng)
        reshaped = x.reshape((2,) * (m + n))
        put(f"{name}/forward_generic", basis.code(*tensors, reshaped))
        put(f"{name}/inverse_generic", basis.inv_code(*tensors, reshaped))
        k = max(1, x.size // 3)
        for label, loss in (("l1", pdft.L1Norm()), ("mse", pdft.MSELoss(k=k))):

            def f(ts, loss=loss):
                return loss_function(ts, m, n, basis.code, x, loss, inverse_code=basis.inv_code)

            put(f"{name}/{label}", f(list(basis.tensors)))
            for where, ts in (("init", list(basis.tensors)), ("generic", tensors)):
                for i, g in enumerate(jax.grad(f)(ts)):
                    put(f"{name}/{label}_grad_{where}{i}", g)
        put(f"{name}/dense_operator", dense_operator(basis))
        put(f"{name}/mu", coherence(basis))
        put(f"{name}/diagonal", np.asarray(diagonal_tensor_indices(basis), dtype=np.int64))
        cert = certify_flat_modulus(basis)
        put(f"{name}/certificate", [float(cert.holds), cert.mu, *cert.offending_indices])
        leaves, treedef = jax.tree_util.tree_flatten(basis)
        put(f"{name}/n_leaves", len(leaves))
        again = jax.tree_util.tree_unflatten(treedef, [2 * leaf for leaf in leaves])
        put(f"{name}/forward_doubled", again.forward_transform(x))

    # 2. the single-image trainer
    for name, (label, loss, optimizer) in itertools.product(
        (
            "qft_3x2",
            "rich_3x2",
            "tebd_u4_2x2",
            "dct4_controlled_3x2",
            "entangled_front_2x3",
            "blocked_qft",
        ),
        (
            ("gd_l1", pdft.L1Norm(), pdft.RiemannianGD(lr=0.01)),
            ("gd_mse", pdft.MSELoss(k=5), pdft.RiemannianGD(lr=0.05, max_ls_steps=3)),
            ("adam_l1", pdft.L1Norm(), pdft.RiemannianAdam(lr=0.01)),
            ("adam_mse_clip", pdft.MSELoss(k=5), pdft.RiemannianAdam(lr=0.02, max_grad_norm=0.3)),
        ),
    ):
        basis = BASES[name]()
        target = jnp.asarray(cimage(rng_for("single" + name), basis.image_size))
        result = pdft.train_basis(basis, target=target, loss=loss, optimizer=optimizer, steps=5)
        put(f"single/{name}/{label}/loss", result.loss_history)
        for i, t in enumerate(result.basis.tensors):
            put(f"single/{name}/{label}/tensor{i}", t)

    # 3. the batched trainer
    CONFIGS = {
        "adam": {"optimizer": "adam", "epochs": 3, "batch_size": 2},
        "gd": {"optimizer": "gd", "epochs": 2, "batch_size": 2},
        "adam_instance": {
            "optimizer": pdft.RiemannianAdam(lr=0.3, beta1=0.8, max_grad_norm=0.5),
            "epochs": 2,
            "batch_size": 3,
        },
        "gd_instance": {
            "optimizer": pdft.RiemannianGD(lr=0.3, armijo_tau=0.3, max_ls_steps=4),
            "epochs": 2,
            "batch_size": 3,
        },
        "adam_val": {
            "optimizer": "adam",
            "epochs": 4,
            "batch_size": 2,
            "validation_split": 0.4,
            "early_stopping_patience": 1,
        },
        "gd_val": {
            "optimizer": "gd",
            "epochs": 3,
            "batch_size": 2,
            "validation_split": 0.2,
            "val_every_k_epochs": 2,
        },
        "adam_frozen": {
            "optimizer": "adam",
            "epochs": 2,
            "batch_size": 5,
            "frozen_indices": [0, 3],
            "max_grad_norm": 0.2,
        },
        "gd_frozen": {
            "optimizer": "gd",
            "epochs": 2,
            "batch_size": 1,
            "frozen_indices": [1],
            "shuffle": False,
        },
        "adam_schedule": {
            "optimizer": "adam",
            "epochs": 2,
            "batch_size": 1,
            "lr_peak": 0.05,
            "lr_final": 0.002,
            "warmup_frac": 0.3,
            "seed": 7,
        },
    }
    for name, (label, config), (loss_label, loss) in itertools.product(
        ("qft_3x2", "rich_3x2", "tebd_u4_2x2", "dct4_controlled_3x2", "mera_cp_4x2", "blocked_qft"),
        CONFIGS.items(),
        (("l1", pdft.L1Norm()), ("mse", pdft.MSELoss(k=5))),
    ):
        basis = BASES[name]()
        rng = rng_for("batched" + name)
        images = [cimage(rng, basis.image_size) for _ in range(5)]
        result = pdft.train_basis_batched(basis, dataset=images, loss=loss, **config)
        key = f"batched/{name}/{label}/{loss_label}"
        put(f"{key}/loss", result.loss_history)
        put(f"{key}/val", result.val_history)
        put(f"{key}/counts", [result.steps, result.epochs_completed])
        for i, t in enumerate(result.basis.tensors):
            put(f"{key}/tensor{i}", t)

    # 4. freezing
    for name in ("qft_3x2", "rich_3x2", "real_rich_2x2"):
        basis = BASES[name]()
        for blocks in ((0, 0), (1, 0), (1, 1)):
            if basis.m - blocks[0] < 1 or basis.n - blocks[1] < 1:
                continue
            frozen, indices = pdft.freeze_as_blocked(basis, *blocks)
            put(f"freeze/{name}/{blocks}/indices", np.asarray(indices, dtype=np.int64))
            x = jnp.asarray(cimage(rng_for("freeze" + name), basis.image_size))
            put(f"freeze/{name}/{blocks}/forward", frozen.forward_transform(x))
            for i, t in enumerate(frozen.tensors):
                put(f"freeze/{name}/{blocks}/tensor{i}", t)

    # 5. serialisation and compression
    for name in ("qft_2x2", "qft_3x2", "qft_1x3"):
        basis = BASES[name]()
        trained = pdft.train_basis(
            basis,
            target=jnp.asarray(cimage(rng_for("io" + name), basis.image_size)),
            loss=pdft.L1Norm(),
            optimizer=pdft.RiemannianAdam(lr=0.05),
            steps=3,
        ).basis
        for label, b in (("init", basis), ("trained", trained)):
            put(f"io/{name}/{label}/json", json.dumps(basis_to_dict(b), sort_keys=True))
            put(f"io/{name}/{label}/hash", basis_hash(b))
            image = rng_for("compress" + name).standard_normal(basis.image_size)
            put(
                f"io/{name}/{label}/ratio",
                json.dumps(compressed_to_dict(compress(b, image, ratio=0.6)), sort_keys=True),
            )
            put(
                f"io/{name}/{label}/k",
                json.dumps(compressed_to_dict(compress_with_k(b, image, k=5)), sort_keys=True),
            )

    # 6. fit_to_dct, with what it prints
    for name, factory in (
        ("rich", lambda: pdft.RichBasis(1, 2)),
        ("real_rich", lambda: pdft.RealRichBasis(2, 1)),
    ):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            tensors = fit_to_dct(factory, n_steps=201, lr=0.02)
        for i, t in enumerate(tensors):
            put(f"fit_to_dct/{name}/tensor{i}", t)
        put(
            f"fit_to_dct/{name}/printed",
            re.sub(r"elapsed [0-9.]+s", "elapsed", captured.getvalue()),
        )

    # 7. single-precision tensors, images of other shapes, keyword calls: dtypes, values, exception types
    def outcome(call):
        try:
            value = call()
        except Exception as error:  # the type of failure is the behaviour being compared
            return f"raises {type(error).__name__}"
        return f"{np.asarray(value).dtype} {np.asarray(value).shape}"

    for name in (
        "qft_3x2",
        "entangled_3x2",
        "tebd_u4_2x2",
        "mera_cp_4x2",
        "dct4_controlled_3x2",
        "rich_3x2",
        "real_rich_2x2",
        "blocked_qft",
        "blocked_rich",
    ):
        basis = BASES[name]()
        rng = rng_for("single precision" + name)
        leaves, treedef = jax.tree_util.tree_flatten(basis)
        single = jax.tree_util.tree_unflatten(
            treedef, [leaf.astype(jnp.complex64) for leaf in leaves]
        )
        x = cimage(rng, basis.image_size)
        images = {
            "f32": jnp.asarray(x.real, dtype=jnp.float32),
            "f64": jnp.asarray(x.real),
            "c64": jnp.asarray(x, dtype=jnp.complex64),
            "c128": jnp.asarray(x),
        }
        m, n = basis.m, basis.n
        for label, image in images.items():
            for which, b in (("double", basis), ("single", single)):
                key = f"precision/{name}/{which}/{label}"
                put(f"{key}/forward", b.forward_transform(image))
                put(f"{key}/inverse", b.inverse_transform(image))
                for loss_label, loss in (("l1", pdft.L1Norm()), ("mse", pdft.MSELoss(k=5))):

                    def f(ts, b=b, image=image, loss=loss):
                        return loss_function(ts, m, n, b.code, image, loss, inverse_code=b.inv_code)

                    put(f"{key}/{loss_label}", f(list(b.tensors)))
                    for i, g in enumerate(jax.grad(f)(list(b.tensors))):
                        put(f"{key}/{loss_label}_grad{i}", g)
        for label in ("f32", "c64", "c128"):
            for opt_label, optimizer in (
                ("gd", pdft.RiemannianGD(lr=0.01)),
                ("adam", pdft.RiemannianAdam(lr=0.01)),
            ):
                result = pdft.train_basis(
                    single, target=images[label], loss=pdft.L1Norm(), optimizer=optimizer, steps=3
                )
                put(f"precision/{name}/train/{label}/{opt_label}/loss", result.loss_history)
                for i, t in enumerate(result.basis.tensors):
                    put(f"precision/{name}/train/{label}/{opt_label}/tensor{i}", t)
        rows, cols = basis.image_size
        shapes = {
            "transposed": jnp.ones((cols, rows)),
            "flat": jnp.ones((rows * cols,)),
            "extra_axis": jnp.ones((1, rows, cols)),
            "too_small": jnp.ones((rows, cols // 2)),
            "stack": jnp.ones((2, rows, cols)),
        }
        for label, image in shapes.items():
            put(f"shapes/{name}/{label}/forward", outcome(lambda: basis.forward_transform(image)))
            put(f"shapes/{name}/{label}/inverse", outcome(lambda: basis.inverse_transform(image)))
            put(
                f"shapes/{name}/{label}/loss",
                outcome(
                    lambda: loss_function(
                        list(basis.tensors), m, n, basis.code, image, pdft.L1Norm()
                    )
                ),
            )

    from pdft.bases.circuit import entangled_qft, mera, tebd

    def capture(call):
        try:
            return np.asarray(call())
        except Exception as error:
            return np.asarray(f"raises {type(error).__name__}")

    kb = pdft.TEBDBasis(m=2, n=2, seed=4)
    put(
        "keywords/tebd_indices",
        capture(lambda: tebd.get_tebd_gate_indices(tensors=kb.tensors, n_gates=4)),
    )
    put(
        "keywords/tebd_phases",
        capture(lambda: tebd.extract_tebd_phases(tensors=kb.tensors, gate_indices=[4, 5])),
    )
    km = pdft.MERABasis(m=2, n=2, seed=4)
    put(
        "keywords/mera_indices",
        capture(lambda: mera.get_mera_gate_indices(tensors=km.tensors, n_gates=4)),
    )
    put(
        "keywords/mera_phases",
        capture(lambda: mera.extract_mera_phases(tensors=km.tensors, gate_indices=[4, 5])),
    )
    ke = pdft.EntangledQFTBasis(m=2, n=2, seed=4)
    put(
        "keywords/entangle_indices",
        capture(
            lambda: entangled_qft.get_entangle_tensor_indices(tensors=ke.tensors, n_entangle=2)
        ),
    )
    put(
        "keywords/entangle_phases",
        capture(
            lambda: entangled_qft.extract_entangle_phases(
                tensors=ke.tensors, entangle_indices=[6, 7]
            )
        ),
    )

    def message(call):
        try:
            call()
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return "no error"

    put("errors/dct4_two_wrong", message(lambda: pdft.DCT4Basis(0, 2, parametrization="zz")))
    put("errors/qft_size", message(lambda: pdft.QFTBasis(0, 2)))
    put("errors/tebd_parametrization", message(lambda: pdft.TEBDBasis(2, 2, parametrization="zz")))
    put(
        "errors/entangle_position",
        message(lambda: pdft.EntangledQFTBasis(2, 2, entangle_position="middle")),
    )
    put("errors/mera_power", message(lambda: pdft.MERABasis(3, 2)))
    put("errors/freeze_tebd", message(lambda: pdft.freeze_as_blocked(pdft.TEBDBasis(2, 2), 1, 1)))
    put(
        "errors/freeze_partition",
        message(lambda: pdft.freeze_as_blocked(pdft.QFTBasis(2, 2), 2, 0)),
    )
    put(
        "errors/mse_without_inverse",
        message(
            lambda: loss_function(
                list(pdft.QFTBasis(2, 2).tensors),
                2,
                2,
                pdft.QFTBasis(2, 2).code,
                jnp.ones((4, 4)),
                pdft.MSELoss(k=3),
            )
        ),
    )
    return out


if __name__ == "__main__":
    records = collect()
    np.savez(sys.argv[1], **records)
    print(f"{len(records)} arrays written to {sys.argv[1]}")
