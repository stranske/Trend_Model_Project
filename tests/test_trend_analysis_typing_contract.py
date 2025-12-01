"""Tests for the :mod:`trend_analysis.typing` module contract."""

from collections import UserDict
from types import MappingProxyType
from typing import (
    Any,
    Mapping,
    MutableMapping,
    MutableSequence,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from trend_analysis.typing import (
    CovarianceDiagonal,
    MultiPeriodPeriodResult,
    StatsMapping,
)


def test_multi_period_period_result_schema_matches_expected_contract() -> None:
    """Verify the ``MultiPeriodPeriodResult`` TypedDict schema."""

    hints = get_type_hints(MultiPeriodPeriodResult)

    # The schema should allow partial dictionaries so downstream code can build
    # results incrementally while still benefiting from type checking.
    assert MultiPeriodPeriodResult.__total__ is False

    assert hints["period"] == tuple[str, str, str, str]
    assert hints["missing_policy_applied"] is bool

    def assert_mapping_union(value: object) -> None:
        typed_value = cast(Any, value)
        origin = get_origin(typed_value)
        assert origin is not None
        assert origin is Union

        args = get_args(typed_value)
        assert Mapping[str, float] in args
        assert MutableMapping[str, float] in args

    assert_mapping_union(hints["out_ew_stats"])
    assert_mapping_union(hints["out_user_stats"])
    assert_mapping_union(hints["cache_stats"])

    assert hints["manager_changes"] == MutableSequence[dict[str, object]]
    assert hints["turnover"] is float
    assert hints["transaction_cost"] is float
    cov_diag_hint = cast(Any, hints["cov_diag"])
    # ``TypeAlias`` annotations resolve to ``list[float]`` at runtime on
    # supported Python versions.  Comparing directly against the specialised
    # ``list`` keeps the assertion type-friendly while remaining precise about
    # the expected element type.
    # Confirm the alias resolves to a parametrised list whose single argument is ``float``.
    # Identity comparison is brittle across Python implementations; structural check instead.
    from typing import get_args as _ga
    from typing import get_origin as _go

    assert _go(cov_diag_hint) is list
    args = _ga(cov_diag_hint)
    assert len(args) == 1 and args[0] is float
    assert _go(CovarianceDiagonal) is list and _ga(CovarianceDiagonal) == (float,)

    stats_mapping_hint = cast(Any, hints["out_ew_stats"])
    assert stats_mapping_hint == StatsMapping


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


def test_multi_period_period_result_accepts_mapping_variants() -> None:
    """The union fields should accept both mutable and immutable mappings."""

    result: MultiPeriodPeriodResult = MultiPeriodPeriodResult()

    # ``period`` is optional because the TypedDict is marked ``total=False``.
    result["period"] = ("2021-01-01", "2021-03-31", "2021-04-01", "2021-06-30")

    # ``MappingProxyType`` provides an immutable ``Mapping`` implementation,
    # while ``UserDict`` exercises the ``MutableMapping`` branch of the union.
    ew_stats = MappingProxyType({"alpha": 1.23})
    user_stats = UserDict({"beta": 4.56})

    result["out_ew_stats"] = ew_stats
    result["out_user_stats"] = user_stats
    result["cache_stats"] = user_stats

    assert result["out_ew_stats"]["alpha"] == 1.23

    # ``UserDict`` remains mutable after assignment, confirming that
    # ``MutableMapping`` values are stored by reference.
    user_stats["beta"] = 7.89
    assert result["out_user_stats"]["beta"] == 7.89
    assert result["cache_stats"]["beta"] == 7.89
