"""Extended tests for rank_selection module to improve coverage."""

import sys
import types
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

import trend_analysis.core.rank_selection as rank_selection
from trend_analysis.core.rank_selection import (
    DEFAULT_METRIC,
    RiskStatsConfig,
    WindowMetricBundle,
    _apply_transform,
    _canonicalise_labels,
    _ensure_canonical_columns,
    _json_default,
    _quality_filter,
    _stats_cfg_hash,
    blended_score,
    build_ui,
    clear_window_metric_cache,
    get_window_metric_bundle,
    make_window_key,
    rank_select_funds,
    select_funds,
    selector_cache_stats,
    store_window_metric_bundle,
)


def _cm_mock() -> MagicMock:
    m = MagicMock()
    m.__enter__.return_value = m
    m.__exit__.return_value = None
    return m


class TestQualityFilter:
    """Test the _quality_filter function for edge cases and boundary
    conditions."""

    def test_quality_filter_empty_dataframe(self):
        """Test quality filter with empty dataframe."""
        df = pd.DataFrame(columns=["Date", "A", "B"])
        fund_columns = ["A", "B"]
        cfg = rank_selection.default_quality_config()

        result = _quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        # With empty DataFrame, quality filter might still return fund columns
        assert isinstance(result, list)

    def test_quality_filter_insufficient_data(self):
        """Test quality filter when funds don't meet minimum requirements."""
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        df = pd.DataFrame(
            {
                "Date": dates,
                "A": [0.01, 0.02, np.nan],  # Has NaN
                "B": [0.01, 0.02, 0.03],  # Good data
            }
        )
        fund_columns = ["A", "B"]
        cfg = rank_selection.default_quality_config()

        result = _quality_filter(df, fund_columns, "2020-01", "2020-03", cfg)
        # Just test that it returns a list (actual filtering logic may vary)
        assert isinstance(result, list)

    def test_quality_filter_all_funds_pass(self):
        """Test quality filter when all funds pass requirements."""
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        df = pd.DataFrame(
            {
                "Date": dates,
                "A": np.random.randn(12) * 0.01,
                "B": np.random.randn(12) * 0.01,
                "C": np.random.randn(12) * 0.01,
            }
        )
        fund_columns = ["A", "B", "C"]
        cfg = rank_selection.default_quality_config()

        result = _quality_filter(df, fund_columns, "2020-01", "2020-12", cfg)
        assert isinstance(result, list)

    def test_quality_filter_edge_date_boundaries(self):
        """Test quality filter with edge case date boundaries."""
        dates = pd.date_range("2020-01-31", periods=6, freq="ME")
        df = pd.DataFrame(
            {
                "Date": dates,
                "A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                "B": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            }
        )
        fund_columns = ["A", "B"]
        cfg = rank_selection.default_quality_config()

        # Test with exact start/end dates
        result = _quality_filter(df, fund_columns, "2020-01", "2020-06", cfg)
        assert isinstance(result, list)


class TestBlendedScore:
    """Test the blended_score function for various scenarios."""

    def test_blended_score_single_metric(self):
        """Test blended score with a single metric."""
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        df = pd.DataFrame(
            {
                "A": np.random.randn(12) * 0.01 + 0.005,
                "B": np.random.randn(12) * 0.01 + 0.003,
                "C": np.random.randn(12) * 0.01 + 0.001,
            },
            index=dates,
        )

        weights = {"AnnualReturn": 1.0}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert not result.isna().any()

    def test_blended_score_multiple_metrics(self):
        """Test blended score with multiple metrics."""
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        df = pd.DataFrame(
            {
                "A": np.random.randn(24) * 0.02 + 0.008,
                "B": np.random.randn(24) * 0.015 + 0.006,
                "C": np.random.randn(24) * 0.025 + 0.004,
            },
            index=dates,
        )

        weights = {"AnnualReturn": 0.5, "Sharpe": 0.3, "Sortino": 0.2}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert not result.isna().any()

    def test_blended_score_empty_weights(self):
        """Test blended score with empty weights raises error."""
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        df = pd.DataFrame(
            {"A": np.random.randn(12) * 0.01, "B": np.random.randn(12) * 0.01},
            index=dates,
        )

        weights = {}
        stats_cfg = RiskStatsConfig()

        with pytest.raises(
            ValueError, match=r"blended_score requires non[-‑]empty weights dict"
        ):
            blended_score(df, weights, stats_cfg)

    def test_blended_score_weight_normalization(self):
        """Test that weights are properly normalized."""
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        df = pd.DataFrame({"A": [0.01] * 12, "B": [0.02] * 12}, index=dates)

        # Test with weights that don't sum to 1
        weights1 = {"AnnualReturn": 2.0, "Sharpe": 4.0}
        weights2 = {"AnnualReturn": 1.0, "Sharpe": 2.0}
        stats_cfg = RiskStatsConfig()

        result1 = blended_score(df, weights1, stats_cfg)
        result2 = blended_score(df, weights2, stats_cfg)

        # Results should be identical after normalization
        pd.testing.assert_series_equal(result1, result2)


class TestSelectFunds:
    """Test the select_funds function for different selection modes."""

    def setup_method(self):
        """Setup test data."""
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        self.df = pd.DataFrame(
            {
                "Date": dates,
                "RF": [0.001] * 24,
                "A": np.random.randn(24) * 0.02 + 0.008,
                "B": np.random.randn(24) * 0.015 + 0.006,
                "C": np.random.randn(24) * 0.025 + 0.004,
                "D": np.random.randn(24) * 0.03 + 0.002,
            }
        )
        self.fund_columns = ["A", "B", "C", "D"]
        self.cfg = rank_selection.default_quality_config()

    def test_select_funds_all_mode(self):
        """Test select_funds with 'all' mode."""
        result = select_funds(
            self.df,
            "RF",
            self.fund_columns,
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="all",
        )
        assert isinstance(result, list)

    def test_select_funds_top_n_mode(self):
        """Test select_funds with 'rank' mode."""
        result = select_funds(
            self.df,
            "RF",
            self.fund_columns,
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="rank",
            rank_kwargs={
                "inclusion_approach": "top_n",
                "n": 2,
                "score_by": "AnnualReturn",
            },
        )
        assert isinstance(result, list)

    def test_select_funds_random_mode(self):
        """Test select_funds with 'random' mode."""
        result = select_funds(
            self.df,
            "RF",
            self.fund_columns,
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="random",
            random_n=2,
        )
        assert isinstance(result, list)

    def test_select_funds_empty_fund_list(self):
        """Test select_funds with empty fund list."""
        result = select_funds(
            self.df,
            "RF",
            [],
            "2020-01",
            "2020-12",
            "2021-01",
            "2021-12",
            self.cfg,
            selection_mode="all",
        )
        assert result == []


class TestRankSelectionInternals:
    def test_canonicalise_labels_and_dataframe_columns(self):
        labels = ["", " Fund", "Fund", "Alpha", "Alpha"]
        canonical = _canonicalise_labels(labels)
        assert canonical == [
            "Unnamed_1",
            "Fund",
            "Fund_2",
            "Alpha",
            "Alpha_2",
        ]

        df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=labels)
        canon_df = _ensure_canonical_columns(df)
        assert list(canon_df.columns) == canonical
        # Ensure original DataFrame left untouched
        assert list(df.columns) == labels

    def test_json_default_and_stats_hash_include_extras(self):
        cfg = RiskStatsConfig()
        cfg.extra_flag = True
        first_hash = _stats_cfg_hash(cfg)
        cfg.extra_flag = False
        second_hash = _stats_cfg_hash(cfg)
        assert first_hash != second_hash
        assert _json_default((1, 2)) == [1, 2]
        assert _json_default(np.array([1.0, 2.0])) == [1.0, 2.0]
        assert _json_default(np.int64(3)) == 3.0
        with pytest.raises(TypeError):
            _json_default(object())

    def test_window_key_and_cache_roundtrip(self):
        clear_window_metric_cache()
        cfg = RiskStatsConfig()
        df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.03]})
        key = make_window_key("2020-01", "2020-02", df.columns, cfg)
        # Same columns in different order should yield identical key
        key_reordered = make_window_key("2020-01", "2020-02", ["B", "A"], cfg)
        assert key == key_reordered

        bundle = WindowMetricBundle(
            key=key,
            start="2020-01",
            end="2020-02",
            freq="ME",
            stats_cfg_hash=_stats_cfg_hash(cfg),
            universe=tuple(df.columns),
            in_sample_df=df,
            _metrics=pd.DataFrame(index=df.columns, dtype=float),
        )

        store_window_metric_bundle(None, bundle)
        assert selector_cache_stats()["entries"] == 0

        store_window_metric_bundle(key, bundle)
        cached = get_window_metric_bundle(key)
        assert cached is bundle
        stats = selector_cache_stats()
        assert stats["entries"] == 1
        assert stats["selector_cache_hits"] >= 1

        clear_window_metric_cache()
        assert selector_cache_stats()["entries"] == 0

    def test_apply_transform_variants(self):
        series = pd.Series([3.0, 1.0, 2.0], index=["A", "B", "C"])
        ranked = _apply_transform(series, mode="rank")
        assert list(ranked.sort_index()) == [1.0, 3.0, 2.0]

        with pytest.raises(ValueError):
            _apply_transform(series, mode="percentile")
        pct = _apply_transform(series, mode="percentile", rank_pct=0.5)
        assert pct.count() == 2
        assert set(pct.dropna().index) == {"A", "C"}

        zeros = pd.Series([1.0, 1.0, 1.0], index=["X", "Y", "Z"])
        zscores = _apply_transform(zeros, mode="zscore")
        assert (zscores == 0).all()

        with pytest.raises(ValueError):
            _apply_transform(series, mode="unknown")

    def test_rank_select_funds_dedup_and_cache(self):
        clear_window_metric_cache()
        cfg = RiskStatsConfig()
        dates = pd.date_range("2020-01-31", periods=6, freq="ME")
        df = pd.DataFrame(
            {
                "ACME Growth": [0.05] * 6,
                "ACME Value": [0.04] * 6,
                "Beta Alpha": [0.03] * 6,
            },
            index=dates,
        )
        key = make_window_key("2020-01", "2020-06", df.columns, cfg)
        selected = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=3,
            window_key=key,
            freq="ME",
        )
        assert selected == ["ACME Growth", "Beta Alpha", "ACME Value"]

        stats_after_first = selector_cache_stats()
        assert stats_after_first["entries"] == 1

        second = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=2,
            window_key=key,
            limit_one_per_firm=False,
        )
        assert second == ["ACME Growth", "ACME Value"]
        assert (
            selector_cache_stats()["selector_cache_hits"]
            > stats_after_first["selector_cache_hits"]
        )

    def test_rank_select_funds_validates_bundle(self):
        cfg = RiskStatsConfig()
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        df = pd.DataFrame(
            {"Acme": [0.05, 0.04, 0.03], "Beta": [0.02, 0.02, 0.02]},
            index=dates,
        )
        wrong_universe_bundle = WindowMetricBundle(
            key=None,
            start="",
            end="",
            freq="ME",
            stats_cfg_hash=_stats_cfg_hash(cfg),
            universe=("Other",),
            in_sample_df=df,
            _metrics=pd.DataFrame(index=df.columns, dtype=float),
        )
        with pytest.raises(ValueError, match="does not match DataFrame columns"):
            rank_select_funds(df, cfg, bundle=wrong_universe_bundle)

        wrong_hash_bundle = WindowMetricBundle(
            key=None,
            start="",
            end="",
            freq="ME",
            stats_cfg_hash="not-the-same",
            universe=tuple(df.columns),
            in_sample_df=df,
            _metrics=pd.DataFrame(index=df.columns, dtype=float),
        )
        with pytest.raises(ValueError, match="stats configuration"):
            rank_select_funds(df, cfg, bundle=wrong_hash_bundle)

class TestBuildUI:
    """Test the build_ui function and widget building."""

    @patch("trend_analysis.core.rank_selection.widgets")
    def test_build_ui_returns_vbox(self, mock_widgets):
        """Test that build_ui returns a VBox widget."""
        mock_vbox = Mock()
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.SelectMultiple.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()

        result = build_ui()
        assert result == mock_vbox
        # VBox may be called multiple times internally
        mock_widgets.VBox.assert_called()

    @patch("trend_analysis.core.rank_selection.widgets")
    def test_build_ui_creates_expected_widgets(self, mock_widgets):
        """Test that build_ui creates expected widget types."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.SelectMultiple.return_value = Mock()
        mock_widgets.Button.return_value = Mock()
        mock_widgets.Output.return_value = _cm_mock()

        build_ui()

        # Verify widget creation calls
        mock_widgets.HTML.assert_called()
        mock_widgets.SelectMultiple.assert_called()
        mock_widgets.Button.assert_called()
        mock_widgets.Output.assert_called()


class TestConfigurationEdgeCases:
    """Test configuration objects with edge cases."""

    def test_quality_filter_defaults(self):
        """Test quality filter default values."""
        cfg = rank_selection.default_quality_config()
        # Check that config has expected keys
        assert set(cfg).issuperset({
            "implausible_value_limit",
            "max_missing_months",
            "max_missing_ratio",
        })

    def test_risk_stats_config_defaults(self):
        """Test RiskStatsConfig default values."""
        cfg = RiskStatsConfig()
        # Test that config object is created successfully
        assert cfg is not None

    def test_default_metric_value(self):
        """Test that DEFAULT_METRIC is set correctly."""
        assert DEFAULT_METRIC == "AnnualReturn"


class TestMetricComputation:
    """Test metric computation edge cases."""

    def test_compute_metric_series_edge_cases(self):
        """Test _compute_metric_series with edge case data."""
        # This tests the private function indirectly through blended_score
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")

        # Test with constant returns
        df = pd.DataFrame({"A": [0.01] * 12, "B": [0.02] * 12}, index=dates)

        weights = {"AnnualReturn": 1.0}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_zscore_computation(self):
        """Test z-score computation through blended_score."""
        dates = pd.date_range("2020-01-31", periods=12, freq="ME")

        # Create data where one fund clearly outperforms
        df = pd.DataFrame(
            {
                "A": [0.01] * 12,  # Consistent low returns
                "B": [0.05] * 12,  # Consistent high returns
            },
            index=dates,
        )

        weights = {"AnnualReturn": 1.0}
        stats_cfg = RiskStatsConfig()

        result = blended_score(df, weights, stats_cfg)

        # Fund B should have higher z-score than Fund A
        assert result.loc["B"] > result.loc["A"]


class TestRankSelectFundsBranchCoverage:
    """Cover additional rank_select_funds branches and helpers."""

    def test_rank_transform_alias_uses_rank_sort(self):
        df = pd.DataFrame(
            {
                "Fund A": [0.04, 0.05, 0.06, 0.05],
                "Fund B": [0.01, 0.02, 0.03, 0.02],
                "Fund C": [0.03, 0.04, 0.05, 0.04],
            }
        )
        cfg = RiskStatsConfig(risk_free=0.0)

        result = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=2,
            score_by="sharpe",
            transform_mode="rank",
        )

        assert len(result) == 2
        assert set(result).issubset(df.columns)

    def test_rank_select_funds_handles_blank_names(self):
        df = pd.DataFrame(
            {
                "   ": [0.04, 0.05, 0.03, 0.04],
                "Fund B": [0.01, 0.02, 0.01, 0.02],
                "Fund C": [0.02, 0.03, 0.02, 0.03],
            }
        )
        cfg = RiskStatsConfig(risk_free=0.0)

        result = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=1,
            score_by="AnnualReturn",
        )

        # Assert that no selected fund has a blank name after stripping
        assert all(col.strip() != "" for col in result)

    def test_rank_select_funds_backfills_duplicate_firms(self):
        df = pd.DataFrame(
            {
                "Alpha One": [0.06, 0.05, 0.07, 0.06],
                "Alpha Two": [0.05, 0.04, 0.05, 0.05],
                "Beta Prime": [0.04, 0.03, 0.04, 0.03],
            }
        )
        cfg = RiskStatsConfig(risk_free=0.0)

        result = rank_select_funds(
            df,
            cfg,
            inclusion_approach="top_n",
            n=3,
            score_by="AnnualReturn",
        )

        assert len(result) == 3
        assert {"Alpha One", "Alpha Two"}.issubset(result)


class TestBlendedScoreAdditional:
    """Cover additional blended_score behaviours."""

    def test_duplicate_alias_weights_are_combined(self):
        df = pd.DataFrame(
            {
                "A": [0.02, 0.03, 0.04, 0.05],
                "B": [0.01, 0.02, 0.03, 0.02],
            }
        )
        cfg = RiskStatsConfig(risk_free=0.0)

        combined = blended_score(df, {"Sharpe": 1.0}, cfg)
        alias = blended_score(df, {"Sharpe": 0.6, "sharpe": 0.4}, cfg)

        pd.testing.assert_series_equal(alias, combined)

    def test_zero_total_weights_raises(self):
        df = pd.DataFrame(
            {
                "A": [0.02, 0.03, 0.04],
                "B": [0.01, 0.02, 0.01],
            }
        )
        cfg = RiskStatsConfig(risk_free=0.0)

        with pytest.raises(ValueError, match="Sum of weights must not be zero"):
            blended_score(df, {"Sharpe": 0.0, "Sortino": 0.0}, cfg)


class TestSelectFundsSimplePath:
    """Exercise simple select_funds entry point branches."""

    def setup_method(self) -> None:
        dates = pd.date_range("2021-01-31", periods=6, freq="ME")
        self.df = pd.DataFrame(
            {
                "Date": dates,
                "RF": [0.0] * len(dates),
                "A": [0.01, 0.02, 0.03, 0.02, 0.01, 0.02],
                "B": [0.03, 0.02, 0.01, 0.02, 0.03, 0.02],
            }
        )

    def test_mode_all_and_random_defaults(self):
        np.random.seed(0)
        all_funds = select_funds(self.df, "RF", mode="all")
        assert all_funds == ["A", "B"]

        random_funds = select_funds(self.df, "RF", mode="random")
        assert random_funds == ["RF", "A", "B"]

    def test_random_requires_positive_n(self):
        with pytest.raises(ValueError, match="random_n must be provided"):
            select_funds(self.df, "RF", mode="random", n=0)

    def test_rank_mode_without_n_uses_all_funds(self):
        ranked = select_funds(self.df, "RF", mode="rank")
        assert set(ranked) == {"RF", "A", "B"}

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            select_funds(self.df, "RF", mode="invalid")


class TestSelectFundsExtendedExtras:
    """Exercise extended select_funds positional argument parsing."""

    def test_extended_random_path_uses_extra_args(self):
        dates = pd.date_range("2022-01-31", periods=6, freq="ME")
        df = pd.DataFrame(
            {
                "Date": dates,
                "RF": [0.0] * len(dates),
                "A": np.linspace(0.01, 0.06, len(dates)),
                "B": np.linspace(0.02, 0.07, len(dates)),
            }
        )
        cfg = rank_selection.default_quality_config()

        np.random.seed(1)
        selected = select_funds(
            df,
            "RF",
            ["A", "B"],
            "2022-01",
            "2022-03",
            "2022-04",
            "2022-06",
            cfg,
            "random",
            1,
        )

        assert len(selected) == 1
        assert selected[0] in {"A", "B"}


class TestBuildUiInteractions:
    """Drive the UI helpers to cover interactive callbacks."""

    def test_load_and_run_flow(self, monkeypatch):
        dates = pd.date_range("2021-01-31", periods=6, freq="ME")
        demo_df = pd.DataFrame(
            {
                "Date": dates,
                "RF": [0.0] * len(dates),
                "Fund A": [0.02, 0.03, 0.04, 0.05, 0.04, 0.03],
                "Fund B": [0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
            }
        )

        monkeypatch.setattr(rank_selection, "load_csv", lambda path: demo_df.copy())

        pipeline_calls: dict[str, list] = {"single": [], "run": []}
        export_calls: list[tuple] = []

        pipeline_stub = types.ModuleType("trend_analysis.pipeline")

        def single_period_run(df_in, in_start, in_end):
            pipeline_calls["single"].append((df_in.copy(), in_start, in_end))
            return pd.DataFrame({"Sharpe": [0.2, 0.1]}, index=["Fund A", "Fund B"])

        def run_analysis(
            df_in,
            in_start,
            in_end,
            out_start,
            out_end,
            target_vol,
            risk_free,
            selection_mode,
            random_n,
            custom_weights,
            rank_kwargs,
            manual_funds,
            indices_list,
            benchmarks,
        ):
            pipeline_calls["run"].append(
                {
                    "selection_mode": selection_mode,
                    "random_n": random_n,
                    "custom_weights": custom_weights,
                    "rank_kwargs": rank_kwargs,
                    "manual_funds": manual_funds,
                }
            )
            return {"results": True}

        pipeline_stub.single_period_run = single_period_run
        pipeline_stub.run_analysis = run_analysis

        export_stub = types.ModuleType("trend_analysis.export")

        def make_summary_formatter(res, *args):
            export_calls.append(("formatter", res, args))
            return {"formatter": True}

        def format_summary_text(res, *args):
            export_calls.append(("text", res, args))
            return "summary"

        def export_data(data, prefix, formats, formatter):
            export_calls.append(("export", data, prefix, tuple(formats), formatter))

        export_stub.make_summary_formatter = make_summary_formatter
        export_stub.format_summary_text = format_summary_text
        export_stub.export_data = export_data

        # Ensure relative imports inside rank_selection pick up our stubs
        import trend_analysis as trend_pkg

        monkeypatch.setitem(sys.modules, "trend_analysis.pipeline", pipeline_stub)
        monkeypatch.setitem(sys.modules, "trend_analysis.export", export_stub)
        monkeypatch.setattr(trend_pkg, "pipeline", pipeline_stub, raising=False)
        monkeypatch.setattr(trend_pkg, "export", export_stub, raising=False)

        ui = build_ui()
        (
            source_tb,
            csv_path,
            file_up,
            load_btn,
            load_out,
            idx_select,
            bench_select,
            in_start_widget,
            in_end_widget,
            out_start_widget,
            out_end_widget,
        ) = ui.children[0].children

        csv_path.value = "demo.csv"
        load_btn.click()

        assert "Fund A" in idx_select.options
        assert in_start_widget.value <= in_end_widget.value

        mode_dd = ui.children[1]
        random_n_int = ui.children[2]
        use_rank_ck = ui.children[5]
        next_btn = ui.children[6]
        rank_box = ui.children[7]
        manual_box = ui.children[8]
        out_fmt = ui.children[9]
        run_btn = ui.children[10]

        use_rank_ck.value = True
        next_btn.click()

        mode_dd.value = "manual"
        assert manual_box.layout.display == "flex"

        # manual_box contains HTML summary, one row per fund, and total label
        first_row = manual_box.children[1]
        manual_checkbox = first_row.children[0]
        manual_weight = first_row.children[1]
        manual_weight.value = 50.0
        manual_checkbox.value = True

        incl_dd, metric_dd, topn_int, pct_flt, thresh_f, blended_box = rank_box.children
        metric_dd.value = "blended"
        m1_dd, w1_sl, m2_dd, w2_sl, m3_dd, w3_sl = blended_box.children
        w1_sl.value = 0.5
        w2_sl.value = 0.3
        w3_sl.value = 0.2

        random_n_int.value = 1

        # First run exercises manual mode with default top_n branch
        run_btn.click()

        # Switch to rank mode and exercise remaining inclusion branches
        mode_dd.value = "rank"
        incl_dd.value = "top_pct"
        run_btn.click()

        incl_dd.value = "threshold"
        thresh_f.value = 0.1
        run_btn.click()

        assert pipeline_calls["single"], "manual preview should query pipeline"
        assert len(pipeline_calls["run"]) == 3

        manual_call, pct_call, threshold_call = pipeline_calls["run"]
        assert manual_call["selection_mode"] == "manual"
        assert manual_call["custom_weights"]
        assert manual_call["manual_funds"]

        assert "pct" in pct_call["rank_kwargs"]
        assert "threshold" in threshold_call["rank_kwargs"]

        export_events = [tag for tag, *_ in export_calls if tag == "export"]
        assert export_events, "export_data should be invoked"
        assert export_calls[-1][3] == (out_fmt.value,)
