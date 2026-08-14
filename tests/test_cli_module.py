"""Regression guards for retired root-level CLI surfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from trend import cli


def test_root_src_cli_is_removed() -> None:
    """The uninstalled root utility must not return."""
    repository_root = Path(__file__).resolve().parents[1]
    assert not (repository_root / "src" / "cli.py").exists()
    assert importlib.util.find_spec("cli") is None


@pytest.mark.parametrize("retired_command", ["cv"])
def test_retired_root_cli_commands_are_not_supported(
    capsys: pytest.CaptureFixture[str], retired_command: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([retired_command, "--help"])

    captured = capsys.readouterr()
    assert exc.value.code == 2
    assert "invalid choice" in captured.err
