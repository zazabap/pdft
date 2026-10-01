"""The circuit plot draws the gate sequence the basis keeps, not a guess from tensor values."""

import matplotlib

matplotlib.use("Agg")  # headless

import pytest

import pdft
from pdft.viz.circuit import plot_circuit


@pytest.mark.parametrize(
    ("make", "counts"),
    [
        (lambda: pdft.QFTBasis(m=2, n=2), "4 H + 2 CP"),
        (lambda: pdft.RichBasis(m=2, n=2), "4 H + 2 U4"),
        (lambda: pdft.DCT4Basis(m=2, n=1, parametrization="controlled"), "U4"),
        (lambda: pdft.BlockedBasis(pdft.QFTBasis(m=2, n=2), 1, 1), "4 H + 2 CP"),
    ],
)
def test_plot_shows_one_column_per_gate(make, counts):
    basis = make()
    fig = plot_circuit(basis)
    ax = fig.axes[0]
    program = getattr(basis, "program", None) or basis.inner.program
    assert counts in ax.get_title() and type(basis).__name__ in ax.get_title()
    # one marker collection per gate, and a wire per qubit of the circuit drawn
    assert len(ax.collections) == len(program.steps)
    assert len(ax.get_yticks()) == program.m + program.n
    two_qubit = sum(1 for _, qubits in program.steps if len(qubits) == 2)
    assert len(ax.lines) == program.m + program.n + two_qubit
    matplotlib.pyplot.close(fig)


def test_custom_title_and_file(tmp_path):
    path = tmp_path / "circuit.png"
    fig = plot_circuit(pdft.TEBDBasis(m=2, n=2), output_path=path, title="rings")
    assert path.exists() and fig.axes[0].get_title() == "rings"
    matplotlib.pyplot.close(fig)
