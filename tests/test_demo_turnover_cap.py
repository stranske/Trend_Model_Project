from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEMO_PATH = _REPO_ROOT / "examples" / "demo_turnover_cap.py"
_spec = importlib.util.spec_from_file_location("demo_turnover_cap", _DEMO_PATH)
assert _spec is not None and _spec.loader is not None
demo_turnover_cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_turnover_cap)


def test_demo_turnover_cap_delegates_to_trend_cli_main() -> None:
    with patch("trend.cli.main", return_value=0) as trend_main:
        result = demo_turnover_cap.main([])

    assert result == 0
    trend_main.assert_called_once_with(["run", "--config", str(demo_turnover_cap.CONFIG_PATH)])
