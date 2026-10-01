"""Smoke tests for matplotlib-based viz (plot is optional extra)."""

import matplotlib

matplotlib.use("Agg")  # headless


from pdft.bases.base import QFTBasis
from pdft.viz import (
    TrainingHistory,
    ema_smooth,
    plot_training_comparison,
    plot_training_loss,
)


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


def test_save_training_plots_writes_one_file_per_history_and_a_comparison(tmp_path):
    from pdft.viz import save_training_plots

    histories = [
        TrainingHistory(losses=[3.0, 2.0, 1.5], label="plain run"),
        TrainingHistory(losses=[3.0, 1.0, 0.5], label="tuned"),
    ]
    paths = save_training_plots(histories, tmp_path / "plots", filename_prefix="run")
    assert [p.name for p in paths] == ["run_plain_run.png", "run_tuned.png", "run_comparison.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_loss_plots_share_their_axes_and_return_the_figure(tmp_path):
    history = TrainingHistory(losses=[2.0, 1.0, 0.5], label="run")
    single = plot_training_loss(history, title="one", smooth_alpha=0.5)
    both = plot_training_comparison([history, history], title="two")
    for fig, title, lines in ((single, "one", 2), (both, "two", 2)):
        ax = fig.axes[0]
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_title()) == ("step", "loss", title)
        assert len(ax.lines) == lines and ax.get_legend() is not None
    assert not list(tmp_path.iterdir())
    matplotlib.pyplot.close("all")
