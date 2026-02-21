"""Pipeline stage modules for trend analysis."""

from .portfolio import (  # noqa: F401
    _assemble_analysis_output,
    _ComputationStage,
    _compute_stats,
    _compute_weights_and_stats,
    _Stats,
    calc_portfolio_returns,
)
from .preprocessing import (  # noqa: F401
    _build_sample_windows,
    _frequency_label,
    _prepare_input_data,
    _prepare_preprocess_stage,
    _preprocessing_summary,
    _PreprocessStage,
    _WindowStage,
)
from .selection import (  # noqa: F401
    _resolve_risk_free_column,
    _select_universe,
    _SelectionStage,
    single_period_run,
)

__all__ = [
    "_ComputationStage",
    "_PreprocessStage",
    "_SelectionStage",
    "_Stats",
    "_WindowStage",
    "_assemble_analysis_output",
    "_build_sample_windows",
    "_compute_stats",
    "_compute_weights_and_stats",
    "_frequency_label",
    "_prepare_input_data",
    "_prepare_preprocess_stage",
    "_preprocessing_summary",
    "_resolve_risk_free_column",
    "_select_universe",
    "calc_portfolio_returns",
    "single_period_run",
]
