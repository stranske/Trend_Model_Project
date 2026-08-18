from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from examples import demo_turnover_cap


def test_demo_turnover_cap_delegates_to_trend_cli_main() -> None:
    with patch("trend.cli.main", return_value=0) as trend_main:
        result = demo_turnover_cap.main([])

    assert result == 0
    trend_main.assert_called_once_with(
        ["run", "--config", str(demo_turnover_cap.CONFIG_PATH)]
    )
