from __future__ import annotations

import pytest

from trend import cli


def test_trend_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "usage:" in captured.out
    assert "trend" in captured.out
