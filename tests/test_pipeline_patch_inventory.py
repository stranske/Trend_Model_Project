from __future__ import annotations

from pathlib import Path

PIPELINE_PATCH_INVENTORY = {
    "tests/test_pipeline_branch_coverage.py": {
        "test_run_analysis_returns_none_when_windows_empty": [
            "pipeline._prepare_input_data",
        ],
        "test_run_analysis_na_policy_branch": [
            "pipeline._prepare_input_data",
            "pipeline.single_period_run",
            "pipeline.compute_trend_signals",
            "pipeline.compute_constrained_weights",
            "pipeline.realised_volatility",
            "pipeline.build_regime_payload",
            "pipeline.information_ratio",
        ],
        "test_run_analysis_information_ratio_fallback": [
            "pipeline._prepare_input_data",
            "pipeline.single_period_run",
            "pipeline.compute_trend_signals",
            "pipeline.compute_constrained_weights",
            "pipeline.realised_volatility",
            "pipeline.build_regime_payload",
            "pipeline.information_ratio",
        ],
    },
    "tests/test_pipeline_helpers_additional.py": {
        "test_run_analysis_short_circuits": [
            "pipeline._prepare_input_data",
        ],
    },
    "tests/test_pipeline_optional_features.py": {
        "test_run_analysis_does_not_duplicate_existing_avg_corr": [
            "pipeline.single_period_run",
        ],
        "test_run_analysis_benchmark_ir_best_effort": [
            "pipeline.calc_portfolio_returns",
            "pipeline.information_ratio",
        ],
        "test_run_analysis_benchmark_ir_handles_scalar_output": [
            "pipeline.information_ratio",
        ],
        "test_run_analysis_benchmark_ir_handles_scalar_response": [
            "pipeline.information_ratio",
        ],
        "test_run_analysis_benchmark_ir_non_numeric_enrichment": [
            "pipeline.calc_portfolio_returns",
            "pipeline.information_ratio",
        ],
    },
    "tests/test_pipeline_run_cache_fallbacks.py": {
        "test_run_analysis_rank_branch_with_fallbacks": [
            "pipeline._prepare_input_data",
            "pipeline.rank_select_funds",
            "pipeline.get_window_metric_bundle",
            "pipeline.make_window_key",
            "pipeline.compute_constrained_weights",
            "pipeline.compute_trend_signals",
        ],
        "test_run_analysis_risk_window_zero_length": [
            "pipeline._prepare_input_data",
            "pipeline.compute_constrained_weights",
            "pipeline.compute_trend_signals",
        ],
        "test_run_analysis_returns_none_when_no_value_columns": [
            "pipeline._prepare_input_data",
        ],
    },
    "tests/test_pipeline_run_analysis_helpers.py": {
        "test_compute_weights_and_stats_produces_metrics": [
            "pipeline_module.single_period_run",
        ],
        "test_compute_weights_scopes_signal_inputs_to_window": [
            "pipeline_module.single_period_run",
            "pipeline_module.compute_trend_signals",
        ],
        "test_compute_weights_rejects_out_of_window_signal_dates": [
            "pipeline_module.single_period_run",
        ],
        "test_assemble_analysis_output_wraps_success": [
            "pipeline_module.single_period_run",
        ],
    },
    "tests/test_pipeline.py": {
        "test_run_analysis_benchmark_ir_fallback": [
            "pipeline.information_ratio",
        ],
    },
}


def test_pipeline_patch_inventory_matches_source() -> None:
    for rel_path, test_map in PIPELINE_PATCH_INVENTORY.items():
        source = Path(rel_path)
        assert source.exists(), f"Missing test file: {rel_path}"
        content = source.read_text(encoding="utf-8")
        for test_name in test_map:
            assert f"def {test_name}" in content, f"Missing test name: {test_name}"
