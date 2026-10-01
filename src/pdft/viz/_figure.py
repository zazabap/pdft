"""What the plotting helpers share: the optional matplotlib import and saving a figure."""

from __future__ import annotations

from pathlib import Path


def require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:  # pragma: no cover - defensive
        raise ImportError(
            "matplotlib is required for pdft.viz. Install with: pip install pdft[plot]"
        ) from e


def save(fig, output_path: str | Path | None) -> None:
    """Write ``fig`` to ``output_path`` when one is given."""
    if output_path is not None:
        fig.savefig(str(output_path), bbox_inches="tight", dpi=120)
