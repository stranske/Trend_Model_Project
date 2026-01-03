from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analysis.results import build_metadata

from ..diagnostics import PipelineResult, pipeline_success
from ..metrics import (
    annual_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)
from ..portfolio import apply_weight_policy
from ..regimes import build_regime_payload
from ..risk import (
    RiskDiagnostics,
    RiskWindow,
    compute_constrained_weights,
    realised_volatility,
)
from ..signals import TrendSpec, compute_trend_signals
from .preprocessing import _PreprocessStage, _WindowStage
from .selection import _SelectionStage

logger = logging.getLogger("trend_analysis.pipeline")

__all__ = [
    "_ComputationStage",
    "_Stats",
    "_assemble_analysis_output",
    "_compute_stats",
    "_compute_weights_and_stats",
    "calc_portfolio_returns",
]


def _default_avg_corr_handler(
    in_scaled: pd.DataFrame,
    out_scaled: pd.DataFrame,
    fund_cols: list[str],
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    corr_in = in_scaled[fund_cols].corr()
    corr_out = out_scaled[fund_cols].corr()
    n_f = len(fund_cols)
    is_avg_corr: dict[str, float] = {}
    os_avg_corr: dict[str, float] = {}
    denominator = float(n_f - 1) if n_f > 1 else 1.0
    for f in fund_cols:
        in_sum = cast(float, corr_in.loc[f].sum())
        out_sum = cast(float, corr_out.loc[f].sum())
        in_val = (in_sum - 1.0) / denominator
        out_val = (out_sum - 1.0) / denominator
        is_avg_corr[f] = float(in_val)
        os_avg_corr[f] = float(out_val)
    return is_avg_corr, os_avg_corr


avg_corr_handler = _default_avg_corr_handler


@dataclass
class _Stats:
    """Container for performance metrics.

    AvgCorr fields are optional and only populated when the user explicitly
    requests the ``AvgCorr`` metric (Issue #1160). They remain ``None`` to
    preserve backward compatibility and avoid altering column order when the
    feature is not in use.
    """

    cagr: float
    vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    information_ratio: float
    is_avg_corr: float | None = None
    os_avg_corr: float | None = None

    def __eq__(self, other: object) -> bool:  # pragma: no cover - exercised via tests
        if not isinstance(other, _Stats):
            return NotImplemented

        def _equal(a: float | None, b: float | None) -> bool:
            if a is None or b is None:
                return a is b
            if a == b:
                return True
            return math.isnan(a) and math.isnan(b)

        return (
            _equal(self.cagr, other.cagr)
            and _equal(self.vol, other.vol)
            and _equal(self.sharpe, other.sharpe)
            and _equal(self.sortino, other.sortino)
            and _equal(self.max_drawdown, other.max_drawdown)
            and _equal(self.information_ratio, other.information_ratio)
            and _equal(self.is_avg_corr, other.is_avg_corr)
            and _equal(self.os_avg_corr, other.os_avg_corr)
        )


@dataclass(slots=True)
class _ComputationStage:
    weights_series: pd.Series
    risk_diagnostics: RiskDiagnostics
    weight_engine_fallback: dict[str, Any] | None
    weight_engine_diagnostics: dict[str, Any] | None
    turnover_cap: float | None
    in_scaled: pd.DataFrame
    out_scaled: pd.DataFrame
    rf_in: pd.Series
    rf_out: pd.Series
    in_stats: dict[str, _Stats]
    out_stats: dict[str, _Stats]
    out_stats_raw: dict[str, _Stats]
    in_ew_stats: _Stats
    out_ew_stats: _Stats
    out_ew_stats_raw: _Stats
    in_user_stats: _Stats
    out_user_stats: _Stats
    out_user_stats_raw: _Stats
    ew_weights: dict[str, float]
    user_weights: dict[str, float]
    score_frame: pd.DataFrame
    weight_policy: dict[str, Any]
    signal_frame: pd.DataFrame
    effective_signal_spec: TrendSpec


def calc_portfolio_returns(
    weights: NDArray[Any], returns_df: pd.DataFrame
) -> pd.Series:
    """Calculate weighted portfolio returns."""
    return returns_df.mul(weights, axis=1).sum(axis=1)


def _compute_stats(
    df: pd.DataFrame,
    rf: pd.Series,
    *,
    in_sample_avg_corr: dict[str, float] | None = None,
    out_sample_avg_corr: dict[str, float] | None = None,
) -> dict[str, _Stats]:
    # Metrics expect 1D Series; iterating keeps the logic simple for a handful
    # of columns and avoids reshaping into higher-dimensional arrays.
    stats: dict[str, _Stats] = {}
    for col in df:
        key = str(col)
        stats[key] = _Stats(
            cagr=float(annual_return(df[col])),
            vol=float(volatility(df[col])),
            sharpe=float(sharpe_ratio(df[col], rf)),
            sortino=float(sortino_ratio(df[col], rf)),
            max_drawdown=float(max_drawdown(df[col])),
            information_ratio=float(information_ratio(df[col], rf)),
            is_avg_corr=(in_sample_avg_corr or {}).get(col),
            os_avg_corr=(out_sample_avg_corr or {}).get(col),
        )
    return stats


def _compute_weights_and_stats(
    preprocess: _PreprocessStage,
    window: _WindowStage,
    selection: _SelectionStage,
    *,
    target_vol: float | None,
    monthly_cost: float,
    custom_weights: dict[str, float] | None,
    weighting_scheme: str | None,
    constraints: Mapping[str, Any] | None,
    risk_window: Mapping[str, Any] | None,
    previous_weights: Mapping[str, float] | None,
    lambda_tc: float | None,
    max_turnover: float | None,
    signal_spec: TrendSpec | None,
    weight_policy: Mapping[str, Any] | None,
    warmup: int,
    min_floor: float,
    stats_cfg: Any,
    weight_engine_params: Mapping[str, Any] | None,
) -> _ComputationStage:
    fund_cols = selection.fund_cols

    def _enforce_window_bounds(
        frame: pd.DataFrame,
        label: str,
        allowed_start: pd.Timestamp,
        allowed_end: pd.Timestamp,
    ) -> None:
        if frame.empty:
            return
        idx = frame.index
        outside_mask = (idx < allowed_start) | (idx > allowed_end)
        if bool(outside_mask.any()):
            oob_index = idx[outside_mask]
            first = pd.Timestamp(oob_index.min())
            last = pd.Timestamp(oob_index.max())
            msg = (
                f"{label} contain dates outside the active analysis window: "
                f"[{first} → {last}] not within [{allowed_start} → {allowed_end}]"
            )
            raise ValueError(msg)

    _enforce_window_bounds(
        window.in_df,
        label="In-sample returns",
        allowed_start=window.in_start,
        allowed_end=window.in_end,
    )
    _enforce_window_bounds(
        window.out_df,
        label="Out-of-sample returns",
        allowed_start=window.out_start,
        allowed_end=window.out_end,
    )

    def _scoped_signal_inputs() -> pd.DataFrame:
        if not fund_cols:
            return pd.DataFrame(dtype=float)

        allowed_start = window.in_start
        allowed_end = window.out_end

        def _filter_window(frame: pd.DataFrame, *, strict: bool) -> pd.DataFrame:
            outside_mask = (frame.index < allowed_start) | (frame.index > allowed_end)
            if strict and bool(outside_mask.any()):
                first = pd.Timestamp(frame.index[outside_mask].min())
                last = pd.Timestamp(frame.index[outside_mask].max())
                msg = (
                    "Signal inputs contain dates outside the active analysis window: "
                    f"[{first} → {last}] not within [{allowed_start} → {allowed_end}]"
                )
                raise ValueError(msg)

            scoped = frame.loc[
                (frame.index >= allowed_start) & (frame.index <= allowed_end)
            ]
            scoped = scoped.loc[:, ~scoped.columns.duplicated()]
            scoped = scoped.reindex(columns=fund_cols)
            return scoped.astype(float)

        signal_source: pd.DataFrame | None = None
        strict_enforcement = True
        try:
            scoped_cols = [preprocess.date_col, *fund_cols]
            # Copy before ``set_index`` so DataFrame subclasses (see
            # ``tests.test_pipeline_helpers_additional.SignalFrame``) can signal
            # that no usable signal data are available.
            subset = preprocess.df[scoped_cols].copy()
            signal_source = subset.set_index(preprocess.date_col)
            strict_enforcement = False
        except Exception:
            signal_source = None

        if signal_source is None:
            signal_source = (
                pd.concat([window.in_df, window.out_df])
                .sort_index()
                .reindex(columns=fund_cols)
            )

        return _filter_window(signal_source, strict=strict_enforcement)

    custom_weights_input = custom_weights is not None
    weight_engine_used = False
    weight_engine_fallback: dict[str, Any] | None = None
    weight_engine_diagnostics: dict[str, Any] | None = None
    if (
        custom_weights is None
        and weighting_scheme
        and weighting_scheme.lower() != "equal"
    ):
        try:
            from ..plugins import create_weight_engine

            cov = window.in_df[fund_cols].cov()
            engine = create_weight_engine(
                weighting_scheme.lower(), **(weight_engine_params or {})
            )
            w_series = engine.weight(cov).reindex(fund_cols).fillna(0.0)
            custom_weights = {c: float(w_series.get(c, 0.0) * 100.0) for c in fund_cols}
            weight_engine_used = True
            weight_engine_diagnostics = getattr(engine, "diagnostics", None)
            if (
                weight_engine_diagnostics
                and isinstance(weight_engine_diagnostics, Mapping)
                and weight_engine_diagnostics.get("used_safe_mode")
            ):
                safe_mode = weight_engine_diagnostics.get("safe_mode")
                condition_number = weight_engine_diagnostics.get("condition_number")
                condition_threshold = weight_engine_diagnostics.get(
                    "condition_threshold"
                )
                condition_source = weight_engine_diagnostics.get("condition_source")
                raw_condition_number = weight_engine_diagnostics.get(
                    "raw_condition_number"
                )
                shrunk_condition_number = weight_engine_diagnostics.get(
                    "shrunk_condition_number"
                )
                shrinkage_info = weight_engine_diagnostics.get("shrinkage")
                fallback_reason = weight_engine_diagnostics.get(
                    "fallback_reason", "safe_mode"
                )
                weight_engine_fallback = {
                    "engine": str(weighting_scheme),
                    "reason": str(fallback_reason),
                    "safe_mode": safe_mode,
                    "condition_number": condition_number,
                    "condition_threshold": condition_threshold,
                    "condition_source": condition_source,
                }
                if raw_condition_number is not None:
                    weight_engine_fallback["raw_condition_number"] = (
                        raw_condition_number
                    )
                if shrunk_condition_number is not None:
                    weight_engine_fallback["shrunk_condition_number"] = (
                        shrunk_condition_number
                    )
                if isinstance(shrinkage_info, Mapping):
                    weight_engine_fallback["shrinkage"] = dict(shrinkage_info)
                if isinstance(condition_number, (int, float)) and isinstance(
                    condition_threshold, (int, float)
                ):
                    logger.warning(
                        "Weight engine '%s' switched to safe mode '%s' "
                        "(%s condition number %.2e > threshold %.2e).",
                        weighting_scheme,
                        safe_mode,
                        condition_source or "covariance",
                        condition_number,
                        condition_threshold,
                    )
                else:
                    logger.warning(
                        "Weight engine '%s' switched to safe mode '%s'.",
                        weighting_scheme,
                        safe_mode,
                    )
            logger.debug(
                "Successfully created %s weight engine",
                weighting_scheme,
                extra={"weight_engine": weighting_scheme},
            )
        except Exception as e:  # pragma: no cover - exercised via tests
            msg = (
                "Weight engine '%s' failed (%s: %s); falling back to equal weights"
                % (weighting_scheme, type(e).__name__, e)
            )
            logger.warning(msg)
            logger.debug(
                "Weight engine creation failed, falling back to equal weights: %s", e
            )
            weight_engine_fallback = {
                "engine": str(weighting_scheme),
                "error_type": type(e).__name__,
                "error": str(e),
                "logger_level": logging.getLevelName(logger.getEffectiveLevel()),
            }
            custom_weights = None

    if custom_weights is None:
        custom_weights = {c: 100 / len(fund_cols) for c in fund_cols}

    constraints_cfg = constraints or {}
    if not isinstance(constraints_cfg, Mapping):
        constraints_cfg = {}
    long_only = bool(constraints_cfg.get("long_only", True))
    raw_max_weight = constraints_cfg.get("max_weight")
    try:
        max_weight_val = float(raw_max_weight) if raw_max_weight is not None else None
    except (TypeError, ValueError):
        max_weight_val = None
    raw_max_active = constraints_cfg.get("max_active_positions")
    if raw_max_active is None:
        raw_max_active = constraints_cfg.get("max_active")
    try:
        max_active_val = int(raw_max_active) if raw_max_active is not None else None
    except (TypeError, ValueError):
        max_active_val = None
    if max_active_val is not None and max_active_val <= 0:
        max_active_val = None
    raw_group_caps = constraints_cfg.get("group_caps")
    group_caps_map = (
        {str(k): float(v) for k, v in raw_group_caps.items()}
        if isinstance(raw_group_caps, Mapping)
        else None
    )
    raw_groups = constraints_cfg.get("groups")
    groups_map = (
        {str(k): str(v) for k, v in raw_groups.items()}
        if isinstance(raw_groups, Mapping)
        else None
    )

    base_series = pd.Series(
        {c: float(custom_weights.get(c, 0.0)) / 100.0 for c in fund_cols},
        dtype=float,
    )
    if float(base_series.sum()) <= 0:
        base_series = pd.Series(
            np.repeat(1.0 / len(fund_cols), len(fund_cols)),
            index=fund_cols,
            dtype=float,
        )

    negative_assets = base_series[base_series < 0].index.tolist()
    if negative_assets:
        if weight_engine_used:
            source = f"weight engine '{weighting_scheme}'"
        elif custom_weights_input:
            source = "custom weights"
        else:
            source = "base weights"
        action = "clip negatives to zero" if long_only else "allow short allocations"
        logger.info(
            "%s produced %d negative weights; long_only=%s so pipeline will %s.",
            source,
            len(negative_assets),
            long_only,
            action,
        )

    window_cfg = dict(risk_window or {})
    try:
        window_length = int(window_cfg.get("length", len(window.in_df)))
    except (TypeError, ValueError):
        window_length = len(window.in_df)
    if window_length <= 0:
        window_length = max(len(window.in_df), 1)
    decay_mode = str(window_cfg.get("decay", "simple"))
    lambda_value = window_cfg.get("lambda", window_cfg.get("ewma_lambda", 0.94))
    try:
        ewma_lambda = float(lambda_value)
    except (TypeError, ValueError):
        ewma_lambda = 0.94
    window_spec = RiskWindow(
        length=window_length, decay=decay_mode, ewma_lambda=ewma_lambda
    )

    turnover_cap = None
    if max_turnover is not None:
        try:
            mt = float(max_turnover)
        except (TypeError, ValueError):
            mt = None
        if mt is not None and mt > 0:
            turnover_cap = mt

    effective_signal_spec = signal_spec or TrendSpec(
        window=window_spec.length,
        min_periods=None,
        lag=1,
        vol_adjust=False,
        vol_target=None,
        zscore=False,
    )
    signal_inputs = _scoped_signal_inputs()
    if not signal_inputs.empty:
        signal_frame = compute_trend_signals(signal_inputs, effective_signal_spec)
    else:
        signal_frame = pd.DataFrame(dtype=float)

    try:
        weights_series, risk_diagnostics = compute_constrained_weights(
            base_series,
            window.in_df[fund_cols],
            window=window_spec,
            target_vol=target_vol,
            periods_per_year=window.periods_per_year,
            floor_vol=min_floor if min_floor > 0 else None,
            long_only=long_only,
            max_weight=max_weight_val,
            max_active_positions=max_active_val,
            previous_weights=previous_weights,
            lambda_tc=lambda_tc,
            max_turnover=turnover_cap,
            group_caps=group_caps_map,
            groups=groups_map,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Risk controls failed; falling back to base weights: %s", exc, exc_info=True
        )
        weights_series = base_series.copy()
        asset_vol = realised_volatility(
            window.in_df[fund_cols],
            window_spec,
            periods_per_year=window.periods_per_year,
        )
        latest_vol = asset_vol.iloc[-1].reindex(fund_cols)
        latest_vol = latest_vol.ffill().bfill()
        positive = latest_vol[latest_vol > 0]
        fallback_vol = float(positive.min()) if not positive.empty else 1.0
        latest_vol = latest_vol.fillna(fallback_vol)
        if min_floor > 0:
            latest_vol = latest_vol.clip(lower=min_floor)
        if target_vol is None:
            scale_factors = pd.Series(1.0, index=fund_cols, dtype=float)
        else:
            scale_factors = (
                pd.Series(target_vol, index=fund_cols, dtype=float)
                .div(latest_vol)
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )
        scaled_returns = window.in_df[fund_cols].mul(scale_factors, axis=1)
        portfolio_returns = scaled_returns.mul(weights_series, axis=1).sum(axis=1)
        portfolio_vol = realised_volatility(
            portfolio_returns.to_frame("portfolio"),
            window_spec,
            periods_per_year=window.periods_per_year,
        )["portfolio"]
        risk_diagnostics = RiskDiagnostics(
            asset_volatility=asset_vol,
            portfolio_volatility=portfolio_vol,
            turnover=pd.Series(dtype=float, name="turnover"),
            turnover_value=float("nan"),
            scale_factors=scale_factors,
        )

    policy_cfg = dict(weight_policy or {})
    policy_mode = str(policy_cfg.get("mode", policy_cfg.get("policy", "drop"))).lower()
    min_assets_policy = int(policy_cfg.get("min_assets", 1) or 0)

    signal_snapshot: pd.Series | None = None
    if not signal_frame.empty:
        try:
            target_index = (
                window.out_df.index[0]
                if len(window.out_df.index)
                else signal_frame.index[-1]
            )
            aligned = signal_frame.reindex(columns=fund_cols)
            if target_index in aligned.index:
                signal_snapshot = aligned.loc[target_index]
            elif not aligned.empty:
                signal_snapshot = aligned.iloc[-1]
        except Exception:  # pragma: no cover - defensive
            signal_snapshot = None

    weights_series = (
        apply_weight_policy(
            weights_series,
            signal_snapshot,
            mode=policy_mode,
            min_assets=min_assets_policy,
            previous=previous_weights,
        )
        .reindex(fund_cols)
        .fillna(0.0)
    )
    scale_factors = risk_diagnostics.scale_factors.reindex(fund_cols).fillna(0.0)

    in_scaled = window.in_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    out_scaled = window.out_df[fund_cols].mul(scale_factors, axis=1) - monthly_cost
    in_scaled = in_scaled.clip(lower=-1.0)
    out_scaled = out_scaled.clip(lower=-1.0)

    if warmup > 0:
        warmup_in = min(warmup, len(in_scaled))
        warmup_out = min(warmup, len(out_scaled))
        if warmup_in:
            in_scaled.iloc[:warmup_in] = 0.0
        if warmup_out:
            out_scaled.iloc[:warmup_out] = 0.0

    in_scaled = in_scaled.fillna(0.0)
    out_scaled = out_scaled.fillna(0.0)

    rf_in = window.in_df[selection.rf_col]
    rf_out = window.out_df[selection.rf_col]

    want_avg_corr = False
    try:
        reg = getattr(stats_cfg, "metrics_to_run", []) or []
        want_avg_corr = "AvgCorr" in reg
    except Exception:  # pragma: no cover - defensive
        want_avg_corr = False

    is_avg_corr: dict[str, float] | None = None
    os_avg_corr: dict[str, float] | None = None
    if want_avg_corr and len(fund_cols) > 1:
        try:
            is_avg_corr, os_avg_corr = avg_corr_handler(
                in_scaled, out_scaled, fund_cols
            )
        except Exception:  # pragma: no cover - defensive
            is_avg_corr = None
            os_avg_corr = None

    in_stats = _compute_stats(
        in_scaled,
        rf_in,
        in_sample_avg_corr=is_avg_corr,
        out_sample_avg_corr=None,
    )
    out_stats = _compute_stats(
        out_scaled,
        rf_out,
        in_sample_avg_corr=None,
        out_sample_avg_corr=os_avg_corr,
    )
    out_stats_raw = _compute_stats(
        window.out_df[fund_cols],
        rf_out,
        in_sample_avg_corr=None,
        out_sample_avg_corr=os_avg_corr,
    )

    ew_weights = np.repeat(1.0 / len(fund_cols), len(fund_cols))
    ew_w_dict = {c: w for c, w in zip(fund_cols, ew_weights)}
    in_ew = calc_portfolio_returns(ew_weights, in_scaled)
    out_ew = calc_portfolio_returns(ew_weights, out_scaled)
    out_ew_raw = calc_portfolio_returns(ew_weights, window.out_df[fund_cols])

    in_ew_stats = _compute_stats(pd.DataFrame({"ew": in_ew}), rf_in)["ew"]
    out_ew_stats = _compute_stats(pd.DataFrame({"ew": out_ew}), rf_out)["ew"]
    out_ew_stats_raw = _compute_stats(pd.DataFrame({"ew": out_ew_raw}), rf_out)["ew"]

    user_w = weights_series.to_numpy(dtype=float, copy=False)
    user_w_dict = {c: float(weights_series[c]) for c in fund_cols}

    in_user = calc_portfolio_returns(user_w, in_scaled)
    out_user = calc_portfolio_returns(user_w, out_scaled)
    out_user_raw = calc_portfolio_returns(user_w, window.out_df[fund_cols])

    in_user_stats = _compute_stats(pd.DataFrame({"user": in_user}), rf_in)["user"]
    out_user_stats = _compute_stats(pd.DataFrame({"user": out_user}), rf_out)["user"]
    out_user_stats_raw = _compute_stats(pd.DataFrame({"user": out_user_raw}), rf_out)[
        "user"
    ]

    return _ComputationStage(
        weights_series=weights_series,
        risk_diagnostics=risk_diagnostics,
        weight_engine_fallback=weight_engine_fallback,
        weight_engine_diagnostics=weight_engine_diagnostics,
        turnover_cap=turnover_cap,
        in_scaled=in_scaled,
        out_scaled=out_scaled,
        rf_in=rf_in,
        rf_out=rf_out,
        in_stats=in_stats,
        out_stats=out_stats,
        out_stats_raw=out_stats_raw,
        in_ew_stats=in_ew_stats,
        out_ew_stats=out_ew_stats,
        out_ew_stats_raw=out_ew_stats_raw,
        in_user_stats=in_user_stats,
        out_user_stats=out_user_stats,
        out_user_stats_raw=out_user_stats_raw,
        ew_weights=ew_w_dict,
        user_weights=user_w_dict,
        score_frame=selection.score_frame,
        weight_policy=policy_cfg,
        signal_frame=signal_frame,
        effective_signal_spec=effective_signal_spec,
    )


def _assemble_analysis_output(
    preprocess: _PreprocessStage,
    window: _WindowStage,
    selection: _SelectionStage,
    computation: _ComputationStage,
    *,
    benchmarks: Mapping[str, str] | None,
    regime_cfg: Mapping[str, Any] | None,
    target_vol: float | None,
    monthly_cost: float,
    min_floor: float,
) -> PipelineResult:
    fund_cols = selection.fund_cols
    rf_col = selection.rf_col
    benchmark_stats = {}
    benchmark_ir = {}
    out_df = window.out_df
    in_df = window.in_df

    benchmarks = benchmarks or {}
    indices_list = selection.indices_list
    all_benchmarks = benchmarks if benchmarks else {}
    if indices_list:
        index_map = {idx: idx for idx in indices_list}
        index_map.update(benchmarks)
        all_benchmarks = index_map

    out_user = calc_portfolio_returns(
        computation.weights_series.to_numpy(dtype=float, copy=False),
        computation.out_scaled,
    )
    out_user_raw = calc_portfolio_returns(
        computation.weights_series.to_numpy(dtype=float, copy=False), out_df[fund_cols]
    )
    out_ew = calc_portfolio_returns(
        np.repeat(1.0 / len(fund_cols), len(fund_cols)), computation.out_scaled
    )
    out_ew_raw = calc_portfolio_returns(
        np.repeat(1.0 / len(fund_cols), len(fund_cols)), out_df[fund_cols]
    )

    for label, col in all_benchmarks.items():
        if col not in in_df.columns or col not in out_df.columns:
            continue
        benchmark_stats[label] = {
            "in_sample": _compute_stats(
                pd.DataFrame({label: in_df[col]}), computation.rf_in
            )[label],
            "out_sample": _compute_stats(
                pd.DataFrame({label: out_df[col]}), computation.rf_out
            )[label],
        }
        ir_series = information_ratio(computation.out_scaled[fund_cols], out_df[col])
        ir_dict = (
            ir_series.to_dict()
            if isinstance(ir_series, pd.Series)
            else {fund_cols[0]: float(ir_series)}
        )
        try:
            ir_eq = information_ratio(out_ew_raw, out_df[col])
            ir_usr = information_ratio(out_user_raw, out_df[col])
            ir_dict["equal_weight"] = (
                float(ir_eq)
                if isinstance(ir_eq, (float, int, np.floating))
                else float("nan")
            )
            ir_dict["user_weight"] = (
                float(ir_usr)
                if isinstance(ir_usr, (float, int, np.floating))
                else float("nan")
            )
        except Exception:
            pass
        benchmark_ir[label] = ir_dict

    regime_returns_map: dict[str, pd.Series] = {
        "User": out_user.astype(float, copy=False),
        "Equal-Weight": out_ew.astype(float, copy=False),
    }
    regime_payload = build_regime_payload(
        data=preprocess.df,
        out_index=out_df.index,
        returns_map=regime_returns_map,
        risk_free=computation.rf_out,
        config=regime_cfg,
        freq_code=preprocess.freq_summary.target,
        periods_per_year=window.periods_per_year,
    )

    metadata = build_metadata(
        universe=preprocess.value_cols_all,
        selected=fund_cols,
        lookbacks={
            "in_start": window.in_start,
            "in_end": window.in_end,
            "out_start": window.out_start,
            "out_end": window.out_end,
        },
        costs={
            "monthly_cost": monthly_cost,
            "target_vol": target_vol,
            "floor_vol": min_floor if min_floor > 0 else None,
            "max_turnover": computation.turnover_cap,
        },
    )
    metadata["frequency"] = preprocess.frequency_payload
    metadata["missing_data"] = preprocess.missing_payload
    metadata["risk_free_column"] = rf_col
    metadata["indices"] = {
        "requested": selection.requested_indices,
        "accepted": selection.indices_list,
        "missing": selection.missing_indices,
    }

    return pipeline_success(
        {
            "selected_funds": fund_cols,
            "risk_free_column": rf_col,
            "risk_free_source": selection.rf_source,
            "in_sample_scaled": computation.in_scaled,
            "out_sample_scaled": computation.out_scaled,
            "in_sample_stats": computation.in_stats,
            "out_sample_stats": computation.out_stats,
            "out_sample_stats_raw": computation.out_stats_raw,
            "in_ew_stats": computation.in_ew_stats,
            "out_ew_stats": computation.out_ew_stats,
            "out_ew_stats_raw": computation.out_ew_stats_raw,
            "in_user_stats": computation.in_user_stats,
            "out_user_stats": computation.out_user_stats,
            "out_user_stats_raw": computation.out_user_stats_raw,
            "ew_weights": computation.ew_weights,
            "fund_weights": computation.user_weights,
            "benchmark_stats": benchmark_stats,
            "benchmark_ir": benchmark_ir,
            "score_frame": computation.score_frame,
            "weight_engine_fallback": computation.weight_engine_fallback,
            "weight_engine_diagnostics": computation.weight_engine_diagnostics,
            "preprocessing": preprocess.preprocess_info,
            "preprocessing_summary": preprocess.preprocess_info.get("summary"),
            "risk_diagnostics": {
                "asset_volatility": computation.risk_diagnostics.asset_volatility,
                "portfolio_volatility": computation.risk_diagnostics.portfolio_volatility,
                "turnover": computation.risk_diagnostics.turnover,
                "turnover_value": computation.risk_diagnostics.turnover_value,
                "scale_factors": computation.risk_diagnostics.scale_factors,
                "final_weights": computation.weights_series,
            },
            "signal_frame": computation.signal_frame,
            "signal_spec": computation.effective_signal_spec,
            "performance_by_regime": regime_payload.get("table", pd.DataFrame()),
            "regime_labels": regime_payload.get("labels", pd.Series(dtype="string")),
            "regime_labels_out": regime_payload.get(
                "out_labels", pd.Series(dtype="string")
            ),
            "regime_notes": regime_payload.get("notes", []),
            "regime_settings": regime_payload.get("settings", {}),
            "regime_summary": regime_payload.get("summary"),
            "metadata": metadata,
        }
    )
