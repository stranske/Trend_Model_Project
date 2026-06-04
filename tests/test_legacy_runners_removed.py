from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "trend_analysis.run_analysis",
        "trend_analysis.run_multi_analysis",
    ],
)
def test_legacy_runner_modules_gone(module_name: str) -> None:
    module_path = Path("src") / Path(*module_name.split(".")).with_suffix(".py")
    assert not module_path.exists()


def test_legacy_runner_names_not_lazy_exported() -> None:
    init_source = Path("src/trend_analysis/__init__.py").read_text()

    assert '"run_multi_analysis"' not in init_source
