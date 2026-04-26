"""Smoke tests for matplotlib-based viz (plot is optional extra)."""

import matplotlib

matplotlib.use("Agg")  # headless


from pdft.basis import QFTBasis  # noqa: E402
from pdft.viz import TrainingHistory, ema_smooth, plot_training_comparison, plot_training_loss  # noqa: E402


def test_ema_smooth_empty():
    assert ema_smooth([]) == []


def test_ema_smooth_preserves_length():
    out = ema_smooth([1.0, 2.0, 3.0, 4.0], alpha=0.5)
    assert len(out) == 4
    assert out[0] == 1.0


def test_plot_training_loss_writes_png(tmp_path):
    h = TrainingHistory(losses=[3.0, 2.5, 2.0, 1.8, 1.7], label="test")
    p = tmp_path / "loss.png"
    plot_training_loss(h, output_path=p, smooth_alpha=0.3)
    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_training_comparison_multiple_histories(tmp_path):
    a = TrainingHistory(losses=[3.0, 2.0, 1.0], label="gd")
    b = TrainingHistory(losses=[3.0, 2.5, 2.0], label="adam")
    p = tmp_path / "cmp.png"
    plot_training_comparison([a, b], output_path=p)
    assert p.exists()


def test_plot_circuit_renders(tmp_path):
    from pdft.viz.circuit import plot_circuit

    basis = QFTBasis(m=2, n=2)
    p = tmp_path / "circuit.png"
    plot_circuit(basis, output_path=p)
    assert p.exists()
