"""Public configuration types use names that describe their distinct roles."""

from __future__ import annotations

import trend.config_schema as config_schema
import trend_analysis.config as config
import trend_analysis.config.validation as validation
from trend_analysis.config.model import DataSettings


def test_resolved_schema_type_has_no_ambiguous_alias() -> None:
    assert config_schema.ResolvedDataSettings.__name__ == "ResolvedDataSettings"
    assert "ResolvedDataSettings" in config_schema.__all__
    assert "DataSettings" not in config_schema.__all__
    assert not hasattr(config_schema, "DataSettings")
    assert DataSettings.__name__ == "DataSettings"


def test_validation_issue_dto_has_non_exception_name() -> None:
    assert validation.ConfigIssue.__name__ == "ConfigIssue"
    assert config.ConfigIssue is validation.ConfigIssue
    assert "ConfigIssue" in config.__all__
    assert not hasattr(validation, "ValidationError")
    assert "ValidationError" not in config.__all__
