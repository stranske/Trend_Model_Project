"""Shared matplotlib setup for trend reporting surfaces."""

from __future__ import annotations

from typing import Any


def init_matplotlib() -> Any:  # pragma: no cover - thin import/config wrapper
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.edgecolor"] = "white"
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.1
    matplotlib.rcParams["savefig.dpi"] = 160
    matplotlib.rcParams["savefig.transparent"] = False
    matplotlib.rcParams["savefig.format"] = "png"
    return plt


__all__ = ["init_matplotlib"]
