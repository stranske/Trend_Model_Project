"""Provide early feedback when tests run without installing the package."""

from __future__ import annotations

import importlib


def _assert_package_installed() -> None:
    for module in ("trend_analysis", "trend_model", "trend_portfolio_app"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:  # pragma: no cover - configuration guard
            message = (
                "The Trend Model packages are not installed. "
                "Run 'pip install -e .[app]' before executing the test suite."
            )
            raise RuntimeError(message) from exc


_assert_package_installed()
