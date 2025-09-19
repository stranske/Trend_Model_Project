"""Tests for the :mod:`trend_analysis.typing` module contract."""

from collections import UserDict
from typing import Mapping, MutableMapping, MutableSequence, Union, get_args, get_origin, get_type_hints

from trend_analysis.typing import MultiPeriodPeriodResult


def test_multi_period_period_result_schema_matches_expected_contract() -> None:
    """Verify the ``MultiPeriodPeriodResult`` TypedDict schema."""

    hints = get_type_hints(MultiPeriodPeriodResult)

    # The schema should allow partial dictionaries so downstream code can build
    # results incrementally while still benefiting from type checking.
    assert MultiPeriodPeriodResult.__total__ is False

    assert hints["period"] == tuple[str, str, str, str]

    def assert_mapping_union(value: object) -> None:
        origin = get_origin(value)
        assert origin is not None
        assert origin is Union

        args = get_args(value)
        assert Mapping[str, float] in args
        assert MutableMapping[str, float] in args

    assert_mapping_union(hints["out_ew_stats"])
    assert_mapping_union(hints["out_user_stats"])
    assert_mapping_union(hints["cache_stats"])

    assert hints["manager_changes"] == MutableSequence[dict[str, object]]
    assert hints["turnover"] == float
    assert hints["transaction_cost"] == float
    assert hints["cov_diag"] == list[float]


def test_multi_period_period_result_supports_incremental_population() -> None:
    """Ensure the TypedDict behaves like a mutable dictionary at runtime."""

    result: MultiPeriodPeriodResult = MultiPeriodPeriodResult(
        period=("2020-01-01", "2020-03-31", "2020-04-01", "2020-06-30"),
        manager_changes=[],
    )

    # Optional fields should not be present until explicitly populated.
    assert "out_ew_stats" not in result

    result["manager_changes"].append({"added": ["ABC"], "removed": []})
    result["turnover"] = 0.75
    result["cov_diag"] = [0.1, 0.2]

    ew_stats = UserDict({"alpha": 1.0})
    result["out_ew_stats"] = ew_stats
    result["cache_stats"] = {"beta": 2.0}

    assert result["turnover"] == 0.75
    assert result["cov_diag"][1] == 0.2
    assert result["out_ew_stats"]["alpha"] == 1.0
    assert result["cache_stats"]["beta"] == 2.0
