from __future__ import annotations

from trend_analysis.config.validation import ValidationResult
from trend_analysis.tool_layer import ToolLayer


def test_validate_config_reports_missing_sections(tmp_path) -> None:
    tool = ToolLayer()

    result = tool.validate_config({}, base_path=tmp_path)

    assert result.success is True
    assert isinstance(result.data, ValidationResult)
    assert result.data.valid is False
    assert result.data.errors
