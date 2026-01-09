from __future__ import annotations

from pathlib import Path

import pytest

from trend_analysis.config.validation import ValidationResult
from trend_analysis.tool_layer import ToolLayer


def test_validate_config_reports_missing_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    config_root = repo_root / "config"
    config_root.mkdir(parents=True)
    monkeypatch.setenv("TREND_REPO_ROOT", str(repo_root))
    tool = ToolLayer()

    result = tool.validate_config({}, base_path=config_root)

    assert result.success is True
    assert isinstance(result.data, ValidationResult)
    assert result.data.valid is False
    assert result.data.errors
