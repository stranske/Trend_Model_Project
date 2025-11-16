"""Unified report generation for CLI and Streamlit surfaces."""

from __future__ import annotations

import base64
import html
import importlib
import io
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import matplotlib
import pandas as pd


def _init_matplotlib() -> Any:
    matplotlib.use("Agg")
    from matplotlib import pyplot as _pyplot  # noqa: E402

    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.edgecolor"] = "white"
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.1
    matplotlib.rcParams["savefig.dpi"] = 160
    matplotlib.rcParams["savefig.transparent"] = False
    matplotlib.rcParams["savefig.format"] = "png"
    return _pyplot


plt = _init_matplotlib()

from trend_analysis.backtesting import BacktestResult  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trend_model.spec import TrendRunSpec
else:  # pragma: no cover - runtime fallback when trend_model unavailable
    TrendRunSpec = Any


@dataclass(slots=True)
class ReportArtifacts:
    """Container for rendered report assets."""

    html: str
    pdf_bytes: bytes | None
    context: Mapping[str, Any]


def _coerce_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.copy()
    if isinstance(value, Mapping):
        return pd.Series(value).astype(float)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return pd.Series(list(value), dtype=float)
    raise TypeError("Unable to convert value to pandas Series")


def _maybe_series(value: Any) -> pd.Series | None:
    try:
        return _coerce_series(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats_to_dict(stats: Any) -> dict[str, float | None]:
    fields = (
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "information_ratio",
    )
    values: dict[str, float | None] = {field: None for field in fields}
    if stats is None:
        return values
    mapping: Mapping[str, Any] | None = None
    if hasattr(stats, "_asdict"):
        try:
            mapping = cast(Mapping[str, Any], stats._asdict())
        except Exception:
            mapping = None
    if mapping is None and hasattr(stats, "__dict__"):
        mapping = cast(Mapping[str, Any], vars(stats))
    if mapping is None and isinstance(stats, Mapping):
        mapping = stats
    if mapping is not None:
        for field in fields:
            values[field] = _safe_float(mapping.get(field))
        return values
    if isinstance(stats, Sequence) and not isinstance(stats, (str, bytes)):
        for field, value in zip(fields, stats):
            values[field] = _safe_float(value)
    return values


def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return 12.0
    if isinstance(index, pd.DatetimeIndex):
        freq = pd.infer_freq(index)
        if freq:
            code = freq.upper()
            if code.startswith(("A", "Y")):
                return 1.0
            if code.startswith("Q"):
                return 4.0
            if code.startswith("M"):
                return 12.0
            if code.startswith("W"):
                return 52.0
            if code.startswith("D") or code.startswith("B"):
                return 252.0
            try:
                offset = pd.tseries.frequencies.to_offset(freq)
                avg_days = offset.nanos / 86400_000_000_000.0
            except (ValueError, AttributeError, TypeError):
                avg_days = None
        else:
            delta = index.to_series().diff().dropna()
            avg_days = delta.dt.days.mean() if not delta.empty else None
        if avg_days is None:
            return 12.0
        if avg_days <= 1.5:
            return 252.0
        if avg_days <= 7:
            return 52.0
        if avg_days <= 31:
            return 12.0
        if avg_days <= 92:
            return 4.0
        return 1.0
    if isinstance(index, pd.PeriodIndex):
        freq = index.freqstr.upper() if index.freqstr else "M"
        if freq.startswith("A") or freq.startswith("Y"):
            return 1.0
        if freq.startswith("Q"):
            return 4.0
        if freq.startswith("W"):
            return 52.0
        if freq.startswith("D"):
            return 252.0
        return 12.0
    return 12.0


def _drawdown_curve(returns: pd.Series) -> pd.Series:
    filled = 1.0 + returns.fillna(0.0)
    curve = filled.cumprod()
    return curve / curve.cummax() - 1.0


WindowMode = Literal["rolling", "expanding"]


def _coerce_window_mode(value: Any) -> WindowMode:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("exp"):
            return "expanding"
        if lowered.startswith("roll"):
            return "rolling"
    return "rolling"


def _build_backtest(result: Any) -> BacktestResult | None:
    details_obj = getattr(result, "details", None)
    details: Mapping[str, Any] = details_obj if isinstance(details_obj, Mapping) else {}
    portfolio = getattr(result, "portfolio", None)
    if portfolio is None:
        portfolio = details.get("portfolio_equal_weight_combined")
        if portfolio is None:
            portfolio = details.get("portfolio")
    series = _maybe_series(portfolio)
    if series is None or series.empty:
        return None
    series = series.sort_index()
    if isinstance(series.index, pd.PeriodIndex):
        calendar = series.index.to_timestamp()
    elif isinstance(series.index, pd.DatetimeIndex):
        calendar = series.index
    else:
        calendar = pd.DatetimeIndex([])
    equity_curve = (1.0 + series.fillna(0.0)).cumprod()
    drawdown = _drawdown_curve(series)
    turnover_series = None
    final_weights = None
    rolling_sharpe = pd.Series(dtype=float)
    metrics: dict[str, float] = {}
    risk_diag = details.get("risk_diagnostics") if details else None
    if isinstance(risk_diag, Mapping):
        turnover_series = _maybe_series(risk_diag.get("turnover"))
        final_weights = _maybe_series(risk_diag.get("final_weights"))
    if turnover_series is None:
        turnover_series = pd.Series(dtype=float)
    if final_weights is None:
        weights_df = pd.DataFrame(dtype=float)
    else:
        weights_df = pd.DataFrame(
            [final_weights], index=[series.index[-1] if len(series.index) else "latest"]
        )
    index_for_periods: pd.Index
    if isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        index_for_periods = series.index
    else:
        index_for_periods = pd.RangeIndex(len(series))
    periods = _periods_per_year(index_for_periods)
    filled = series.fillna(0.0)
    total_return = float((1.0 + filled).prod() - 1.0)
    n_periods = max(len(filled), 1)
    ann_return = float((1.0 + total_return) ** (periods / n_periods) - 1.0)
    volatility = (
        float(filled.std(ddof=0) * math.sqrt(periods))
        if len(filled.dropna()) > 1
        else 0.0
    )
    annualised_mean = float(filled.mean() * periods)
    sharpe = annualised_mean / (volatility + 1e-12) if volatility else 0.0
    drawdown_min = float(drawdown.min()) if not drawdown.empty else 0.0
    metrics = {
        "total_return": total_return,
        "annual_return": ann_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown_min,
        "turnover_mean": (
            float(turnover_series.mean()) if not turnover_series.empty else 0.0
        ),
    }
    window_mode_value = details.get("window_mode") if details else None
    window_mode = _coerce_window_mode(window_mode_value)
    window_size_value = details.get("window_size") if details else None
    if isinstance(window_size_value, (int, float)):
        window_size = max(int(window_size_value), 1)
    else:
        window_size = max(min(len(series), 120), 1)
    rolling_window = min(len(series), max(int(periods // 4), 1))
    if rolling_window >= 2:
        roll = filled.rolling(rolling_window).mean() / (
            filled.rolling(rolling_window).std(ddof=0) + 1e-12
        )
        rolling_sharpe = roll * math.sqrt(periods)
    return BacktestResult(
        returns=series,
        equity_curve=equity_curve,
        weights=weights_df,
        turnover=turnover_series.sort_index(),
        per_period_turnover=pd.Series(dtype=float),
        transaction_costs=pd.Series(dtype=float),
        rolling_sharpe=rolling_sharpe,
        drawdown=drawdown,
        metrics=metrics,
        calendar=pd.DatetimeIndex(calendar),
        window_mode=window_mode,
        window_size=window_size,
        training_windows={},
    )


def _format_percent(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.1%}"


def _format_ratio(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:.2f}"


def _format_number(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{value:,.2f}"


def _build_exec_summary(result: Any, backtest: BacktestResult | None) -> list[str]:
    details = (
        getattr(result, "details", {})
        if isinstance(getattr(result, "details", None), Mapping)
        else {}
    )
    out_stats = _stats_to_dict(details.get("out_user_stats"))
    ew_stats = _stats_to_dict(details.get("out_ew_stats"))
    selected = details.get("selected_funds")
    manager_count = len(selected) if isinstance(selected, Sequence) else None
    bullets: list[str] = []
    bullets.append(
        (
            "User-weight portfolio delivered an out-of-sample CAGR of "
            f"{_format_percent(out_stats['cagr'])} with Sharpe {_format_ratio(out_stats['sharpe'])} "
            f"and max drawdown {_format_percent(out_stats['max_drawdown'])}."
        )
    )
    bullets.append(
        ("Equal-weight baseline recorded {_cagr} CAGR with Sharpe {_sharpe}.").format(
            _cagr=_format_percent(ew_stats.get("cagr")),
            _sharpe=_format_ratio(ew_stats.get("sharpe")),
        )
    )
    if manager_count is not None:
        bullets.append(
            f"Selection size: {manager_count} funds; diversification assessed below."
        )
    if backtest is not None:
        metrics = backtest.metrics
        bullets.append(
            (
                "Portfolio compounded to {total} total return "
                "({annual} annualised) with realised volatility {vol} "
                "and Sharpe {sharpe}."
            ).format(
                total=_format_percent(metrics.get("total_return")),
                annual=_format_percent(metrics.get("annual_return")),
                vol=_format_percent(metrics.get("volatility")),
                sharpe=_format_ratio(metrics.get("sharpe_ratio")),
            )
        )
    regime_summary = details.get("regime_summary")
    if isinstance(regime_summary, str) and regime_summary.strip():
        bullets.append(regime_summary.strip())
    return bullets


def _extend_params(
    params: list[tuple[str, str]], entries: Sequence[tuple[str, str]]
) -> None:
    existing = {key for key, _ in params}
    for key, value in entries:
        if key in existing:
            continue
        params.append((key, value))


def _trend_spec_summary(spec: Any | None) -> list[tuple[str, str]]:
    if spec is None:
        return []
    entries: list[tuple[str, str]] = []
    entries.append(("Trend window", f"{spec.window} periods"))
    entries.append(("Signal lag", str(spec.lag)))
    if getattr(spec, "min_periods", None):
        entries.append(("Min periods", str(spec.min_periods)))
    scaling = "Vol-adjusted" if spec.vol_adjust else "Raw"
    entries.append(("Signal scaling", scaling))
    target = getattr(spec, "vol_target", None)
    if spec.vol_adjust and isinstance(target, (int, float)):
        entries.append(("Signal vol target", _format_percent(float(target))))
    entries.append(("Signal z-score", "Enabled" if spec.zscore else "Disabled"))
    return entries


def _rank_summary(rank_cfg: Mapping[str, Any]) -> str:
    if not rank_cfg:
        return ""
    approach = str(rank_cfg.get("inclusion_approach", "")) or "unspecified"
    parts = [approach]
    if approach == "top_n" and rank_cfg.get("n") is not None:
        try:
            parts.append(f"n={int(rank_cfg['n'])}")
        except (TypeError, ValueError, KeyError):
            pass
    if approach == "top_pct" and rank_cfg.get("pct") is not None:
        try:
            parts.append(f"pct={float(rank_cfg['pct']):.0%}")
        except (TypeError, ValueError, KeyError):
            pass
    if approach == "threshold" and rank_cfg.get("threshold") is not None:
        try:
            parts.append(f"≥ {float(rank_cfg['threshold']):.2f}")
        except (TypeError, ValueError, KeyError):
            pass
    score_by = rank_cfg.get("score_by")
    if score_by:
        parts.append(f"score={score_by}")
    return ", ".join(parts)


def _backtest_spec_summary(spec: Any | None) -> list[tuple[str, str]]:
    if spec is None:
        return []
    entries: list[tuple[str, str]] = []
    rank_cfg = getattr(spec, "rank", {})
    if isinstance(rank_cfg, Mapping):
        rank_desc = _rank_summary(rank_cfg)
        if rank_desc:
            entries.append(("Rank inclusion", rank_desc))
    metrics = getattr(spec, "metrics", ())
    if metrics:
        metric_text = ", ".join(str(item) for item in metrics)
        entries.append(("Metric registry", metric_text))
    regime = getattr(spec, "regime", {})
    if isinstance(regime, Mapping) and regime:
        enabled = bool(regime.get("enabled", True))
        if not enabled:
            entries.append(("Regime analysis", "Disabled"))
        else:
            method = regime.get("method")
            proxy = regime.get("proxy")
            parts = []
            if method:
                parts.append(str(method))
            if proxy:
                parts.append(f"proxy={proxy}")
            entries.append(("Regime analysis", ", ".join(parts) or "Enabled"))
    multi = getattr(spec, "multi_period", {})
    if isinstance(multi, Mapping) and multi.get("frequency"):
        freq = str(multi.get("frequency"))
        entries.append(("Multi-period frequency", freq))
    return entries


def _build_param_summary(
    config: Any, spec: TrendRunSpec | None = None
) -> list[tuple[str, str]]:
    def _get(section: Any, key: str, default: Any = None) -> Any:
        if section is None:
            return default
        if isinstance(section, Mapping):
            return section.get(key, default)
        return getattr(section, key, default)

    sample = _get(config, "sample_split", {})
    vol_adj = _get(config, "vol_adjust", {})
    portfolio = _get(config, "portfolio", {})
    run_cfg = _get(config, "run", {})
    benchmarks = _get(config, "benchmarks", {})
    params: list[tuple[str, str]] = []
    in_start = _get(sample, "in_start")
    in_end = _get(sample, "in_end")
    out_start = _get(sample, "out_start")
    out_end = _get(sample, "out_end")
    if in_start or in_end:
        params.append(("In-sample window", f"{in_start or '—'} → {in_end or '—'}"))
    if out_start or out_end:
        params.append(
            ("Out-of-sample window", f"{out_start or '—'} → {out_end or '—'}")
        )
    target_vol = _get(vol_adj, "target_vol")
    if isinstance(target_vol, (int, float)):
        params.append(("Target volatility", _format_percent(float(target_vol))))
    floor_vol = _get(vol_adj, "floor_vol")
    if isinstance(floor_vol, (int, float)):
        params.append(("Floor volatility", _format_percent(float(floor_vol))))
    warmup = _get(vol_adj, "warmup_periods")
    if isinstance(warmup, (int, float)):
        params.append(("Warm-up periods", f"{int(warmup)}"))
    selection_mode = _get(portfolio, "selection_mode")
    if selection_mode:
        params.append(("Selection mode", str(selection_mode)))
    weighting = _get(portfolio, "weighting_scheme")
    if weighting:
        params.append(("Weighting scheme", str(weighting)))
    max_turnover = _get(portfolio, "max_turnover")
    if isinstance(max_turnover, (int, float)):
        params.append(("Turnover cap", _format_percent(float(max_turnover))))
    tx_cost = _get(portfolio, "transaction_cost_bps") or _get(run_cfg, "monthly_cost")
    if isinstance(tx_cost, (int, float)):
        params.append(("Transaction cost", f"{float(tx_cost):.2f} bps"))
    rebalance = _get(portfolio, "rebalance_calendar")
    if rebalance:
        params.append(("Rebalance calendar", str(rebalance)))
    bench_count = len(benchmarks) if isinstance(benchmarks, Mapping) else 0
    if bench_count:
        params.append(("Benchmarks", str(bench_count)))
    resolved_spec = spec or getattr(config, "_trend_run_spec", None)
    trend_spec_obj = getattr(resolved_spec, "trend", None) if resolved_spec else None
    if trend_spec_obj is None:
        trend_spec_obj = getattr(config, "trend_spec", None)
    backtest_spec_obj = (
        getattr(resolved_spec, "backtest", None) if resolved_spec else None
    )
    if backtest_spec_obj is None:
        backtest_spec_obj = getattr(config, "backtest_spec", None)
    _extend_params(params, _trend_spec_summary(trend_spec_obj))
    _extend_params(params, _backtest_spec_summary(backtest_spec_obj))
    return params


def _build_caveats(result: Any, backtest: BacktestResult | None) -> list[str]:
    caveats: list[str] = []
    fallback = getattr(result, "fallback_info", None)
    if isinstance(fallback, Mapping) and fallback:
        engine = fallback.get("engine") or "unknown"
        reason = (
            fallback.get("error") or fallback.get("error_type") or "unspecified error"
        )
        caveats.append(f"Weight engine fallback engaged ({engine}): {reason}.")
    details = (
        getattr(result, "details", {})
        if isinstance(getattr(result, "details", None), Mapping)
        else {}
    )
    if not getattr(result, "metrics", pd.DataFrame()).size:
        caveats.append(
            "Metrics table is empty – verify scoring inputs and configuration."
        )
    selected = details.get("selected_funds")
    if isinstance(selected, Sequence) and not selected:
        caveats.append("No funds selected after preprocessing filters.")
    if backtest is None:
        caveats.append(
            "Backtest result unavailable – narrative derived from limited data."
        )
    return caveats


def _render_chart(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": "trend-report"})
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _turnover_chart(backtest: BacktestResult | None) -> str | None:
    if backtest is None or backtest.turnover.empty:
        return None
    series = backtest.turnover.sort_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    x_values = (
        series.index.to_pydatetime()
        if isinstance(series.index, pd.DatetimeIndex)
        else list(series.index)
    )
    ax.plot(x_values, series.to_numpy(dtype=float), color="#1f77b4", linewidth=2)
    ax.set_title("Turnover per rebalance")
    ax.set_ylabel("Turnover")
    ax.grid(alpha=0.25)
    if isinstance(series.index, pd.DatetimeIndex):
        fig.autofmt_xdate()
    return _render_chart(fig)


def _exposure_chart(backtest: BacktestResult | None) -> str | None:
    if backtest is None or backtest.weights.empty:
        return None
    latest = backtest.weights.iloc[-1].sort_values(ascending=False)
    if latest.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(latest.index.astype(str), latest.to_numpy(dtype=float), color="#ff7f0e")
    ax.set_title("Latest portfolio weights")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, max(0.0001, latest.max() * 1.2))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _render_chart(fig)


def _metrics_table_html(metrics: pd.DataFrame) -> tuple[str, list[str]]:
    if metrics.empty:
        return "<p>No metrics available.</p>", []
    display = metrics.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(
                lambda val: "" if pd.isna(val) else f"{val:,.4f}"
            )
        else:
            display[column] = display[column].astype(str)
    display.index = display.index.astype(str)
    html_table = display.to_html(classes=["report-table"], border=0, escape=False)
    text_rows: list[str] = []
    header = ["Name"] + list(display.columns)
    text_rows.append(" | ".join(header))
    text_rows.append("-" * len(text_rows[0]))
    for idx, row in display.iterrows():
        values = [str(idx)] + [str(row[col]) for col in display.columns]
        text_rows.append(" | ".join(values))
    return html_table, text_rows


def _format_regime_table(table: pd.DataFrame) -> tuple[str, list[str]]:
    if table is None or table.empty:
        return "<p>Regime analysis unavailable.</p>", []

    display = table.copy().astype(object)
    idx_names = ["CAGR", "Max Drawdown", "Hit Rate", "Sharpe", "Observations"]
    for metric in display.index:
        if metric not in idx_names:
            continue
        series = display.loc[metric]
        if metric in {"CAGR", "Max Drawdown", "Hit Rate"}:
            display.loc[metric] = series.map(
                lambda val: "—" if pd.isna(val) else f"{float(val):.1%}"
            )
        elif metric == "Sharpe":
            display.loc[metric] = series.map(
                lambda val: "—" if pd.isna(val) else f"{float(val):.2f}"
            )
        elif metric == "Observations":
            display.loc[metric] = series.map(
                lambda val: "—" if pd.isna(val) else f"{int(round(float(val)))}"
            )

    if isinstance(display.columns, pd.MultiIndex):
        column_labels = [" / ".join(map(str, col)).strip() for col in display.columns]
    else:
        column_labels = [str(col) for col in display.columns]

    html_table = display.to_html(classes=["report-table"], border=0, escape=False)
    text_rows: list[str] = []
    header = ["Metric"] + column_labels
    text_rows.append(" | ".join(header))
    text_rows.append("-" * len(text_rows[0]))
    for metric, row in display.iterrows():
        values = [str(metric)] + [str(row[col]) for col in display.columns]
        text_rows.append(" | ".join(values))
    return html_table, text_rows


def _narrative(
    backtest: BacktestResult | None, regime_summary: str | None = None
) -> str:
    if backtest is None or backtest.returns.empty:
        base = (
            "Backtest metrics were unavailable; please review the configuration and ensure "
            "that portfolio returns were produced."
        )
        if regime_summary:
            return f"{base} {regime_summary}".strip()
        return base
    metrics = backtest.metrics
    start = backtest.returns.index[0]
    end = backtest.returns.index[-1]
    if isinstance(start, (pd.Timestamp, datetime)):
        start_text = pd.Timestamp(start).strftime("%b %Y")
    else:
        start_text = str(start)
    if isinstance(end, (pd.Timestamp, datetime)):
        end_text = pd.Timestamp(end).strftime("%b %Y")
    else:
        end_text = str(end)
    top_alloc = (
        backtest.weights.iloc[-1]
        if not backtest.weights.empty
        else pd.Series(dtype=float)
    )
    if not top_alloc.empty:
        top_alloc = top_alloc.sort_values(ascending=False).head(3)
        alloc_text = ", ".join(
            f"{name}: {_format_percent(weight)}" for name, weight in top_alloc.items()
        )
        alloc_sentence = f" Key allocations: {alloc_text}."
    else:
        alloc_sentence = ""
    turnover = metrics.get("turnover_mean", 0.0)
    narrative = (
        f"From {start_text} through {end_text}, the portfolio compounded to "
        f"{_format_percent(metrics.get('total_return'))} overall "
        f"({_format_percent(metrics.get('annual_return'))} annualised). Volatility averaged "
        f"{_format_percent(metrics.get('volatility'))}, driving a Sharpe ratio of "
        f"{_format_ratio(metrics.get('sharpe_ratio'))}. The deepest drawdown reached "
        f"{_format_percent(metrics.get('max_drawdown'))}, and mean turnover per rebalance was "
        f"{_format_percent(turnover)}.{alloc_sentence}"
    )
    if regime_summary:
        narrative = f"{narrative} {regime_summary}".strip()
    return narrative


def _render_html(context: Mapping[str, Any]) -> str:
    title = html.escape(context["title"])
    run_id = html.escape(context["run_id"])
    exec_items = "\n".join(
        f"      <li>{html.escape(item)}</li>" for item in context["exec_summary"]
    )
    narrative = html.escape(context["narrative"])
    metrics_html = context["metrics_html"]
    regime_table_html = context["regime_html"]
    regime_summary_text = context.get("regime_summary") or ""
    regime_summary_html = (
        f"    <p>{html.escape(regime_summary_text)}</p>\n"
        if regime_summary_text
        else ""
    )
    regime_notes = context.get("regime_notes", [])
    regime_notes_html = ""
    if regime_notes:
        items = "\n".join(
            f"      <li>{html.escape(note)}</li>" for note in regime_notes
        )
        regime_notes_html = f"    <ul>\n{items}\n    </ul>\n"
    params_rows = "\n".join(
        f"      <tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
        for k, v in context["parameters"]
    )
    caveats_items = "\n".join(
        f"      <li>{html.escape(item)}</li>" for item in context["caveats"]
    )
    turnover_img = (
        f'<img src="data:image/png;base64,{context["turnover_chart"]}" alt="Turnover chart" />'
        if context["turnover_chart"]
        else "<p>No turnover history available.</p>"
    )
    exposure_img = (
        f'<img src="data:image/png;base64,{context["exposure_chart"]}" alt="Exposure chart" />'
        if context["exposure_chart"]
        else "<p>No weight snapshot available.</p>"
    )
    footer = html.escape(context["footer"])
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        f"  <title>{title}</title>\n"
        "  <style>\n"
        "    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; color: #111; }\n"
        "    h1, h2 { color: #0f172a; }\n"
        "    section { margin-bottom: 2.5rem; }\n"
        "    .report-table { border-collapse: collapse; width: 100%; font-size: 0.95rem; }\n"
        "    .report-table th, .report-table td { border: 1px solid #d1d5db; padding: 0.5rem; text-align: right; }\n"
        "    .report-table th { background: #f1f5f9; text-align: left; }\n"
        "    .two-column { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; }\n"
        "    footer { font-size: 0.85rem; color: #475569; border-top: 1px solid #e2e8f0; padding-top: 1rem; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <header><h1>{title}</h1><p>Run ID: {run_id}</p></header>\n"
        '  <section id="executive-summary">\n'
        "    <h2>Executive summary</h2>\n"
        "    <ul>\n"
        f"{exec_items}\n"
        "    </ul>\n"
        "  </section>\n"
        '  <section id="narrative">\n'
        "    <h2>Narrative</h2>\n"
        f"    <p>{narrative}</p>\n"
        "  </section>\n"
        '  <section id="metrics">\n'
        "    <h2>Metrics</h2>\n"
        f"    {metrics_html}\n"
        "  </section>\n"
        '  <section id="regimes">\n'
        "    <h2>Performance by Regime</h2>\n"
        f"{regime_summary_html}"
        f"    {regime_table_html}\n"
        f"{regime_notes_html}"
        "  </section>\n"
        '  <section id="charts" class="two-column">\n'
        "    <div><h2>Turnover</h2>\n"
        f"    {turnover_img}</div>\n"
        "    <div><h2>Exposures</h2>\n"
        f"    {exposure_img}</div>\n"
        "  </section>\n"
        '  <section id="parameters">\n'
        "    <h2>Parameter summary</h2>\n"
        '    <table class="report-table">\n'
        f"{params_rows}\n"
        "    </table>\n"
        "  </section>\n"
        '  <section id="caveats">\n'
        "    <h2>Caveats</h2>\n"
        "    <ul>\n"
        f"{caveats_items}\n"
        "    </ul>\n"
        "  </section>\n"
        f"  <footer>{footer}</footer>\n"
        "</body>\n"
        "</html>\n"
    )


def _pdf_safe(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\x00", "")
    return cleaned.encode("latin-1", "replace").decode("latin-1")


def _wrap_pdf_text(
    text: str,
    *,
    width: int = 90,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> str:
    """Wrap and format text for PDF output, ensuring lines do not exceed a
    specified width.

    Parameters:
        text (str): The input text to wrap.
        width (int): Maximum line width in characters. Defaults to 90.
        initial_indent (str): String to prepend to the first line. Defaults to "".
        subsequent_indent (str): String to prepend to all subsequent lines. Defaults to "".

    Returns:
        str: The wrapped and formatted text suitable for PDF rendering.
    """
    raw = str(text).replace("\n", " ").strip()
    if not raw:
        return initial_indent.rstrip() if initial_indent else ""
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=True,
        drop_whitespace=True,
    )
    lines = wrapper.wrap(raw)
    return "\n".join(lines) if lines else raw


def _load_fpdf() -> type[Any] | None:
    try:
        module = importlib.import_module("fpdf")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
        return None
    except Exception:  # pragma: no cover - unexpected import failure
        return None
    pdf_cls = getattr(module, "FPDF", None)
    return pdf_cls if isinstance(pdf_cls, type) else None


def _render_pdf(context: Mapping[str, Any]) -> bytes:
    pdf_cls = _load_fpdf()
    if pdf_cls is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDF generation requires the 'fpdf2' package. Install trend-model with PDF extras."
        )
    pdf = pdf_cls()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    usable_width = max(pdf.w - pdf.l_margin - pdf.r_margin, 10)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _pdf_safe(str(context["title"])), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, _pdf_safe(f"Run ID: {context['run_id']}"), ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Executive summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for item in context["exec_summary"]:
        wrapped = _wrap_pdf_text(
            item, initial_indent="- ", subsequent_indent="  ", width=84
        )
        pdf.multi_cell(usable_width, 6, _pdf_safe(wrapped))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Narrative", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        usable_width,
        6,
        _pdf_safe(_wrap_pdf_text(context["narrative"], width=84)),
    )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Metrics", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for row in context["metrics_text"]:
        pdf.multi_cell(usable_width, 5, _pdf_safe(_wrap_pdf_text(row, width=84)))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Performance by Regime", ln=True)
    pdf.set_font("Helvetica", "", 10)
    regime_summary = context.get("regime_summary")
    if regime_summary:
        pdf.multi_cell(
            usable_width,
            5,
            _pdf_safe(_wrap_pdf_text(str(regime_summary), width=84)),
        )
    for row in context.get("regime_text", []):
        pdf.multi_cell(usable_width, 5, _pdf_safe(_wrap_pdf_text(row, width=84)))
    regime_notes = context.get("regime_notes", [])
    for note in regime_notes:
        wrapped_note = _wrap_pdf_text(
            note, initial_indent="- ", subsequent_indent="  ", width=84
        )
        pdf.multi_cell(usable_width, 5, _pdf_safe(wrapped_note))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Parameter summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for key, value in context["parameters"]:
        pdf.multi_cell(
            usable_width,
            5,
            _pdf_safe(
                _wrap_pdf_text(
                    f"{key}: {value}",
                    width=84,
                    initial_indent="",
                    subsequent_indent="    ",
                )
            ),
        )
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Caveats", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for caveat in context["caveats"]:
        wrapped_caveat = _wrap_pdf_text(
            caveat, initial_indent="- ", subsequent_indent="  ", width=84
        )
        pdf.multi_cell(usable_width, 5, _pdf_safe(wrapped_caveat))
    if context["turnover_chart"]:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Turnover", ln=True)
        chart_bytes = base64.b64decode(context["turnover_chart"])
        pdf.image(io.BytesIO(chart_bytes), x=pdf.l_margin, w=180)
    if context["exposure_chart"]:
        if not context["turnover_chart"]:
            pdf.add_page()
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Exposures", ln=True)
        chart_bytes = base64.b64decode(context["exposure_chart"])
        pdf.image(io.BytesIO(chart_bytes), x=pdf.l_margin, w=180)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        usable_width,
        5,
        _pdf_safe(_wrap_pdf_text(context["footer"], width=84)),
    )
    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    if isinstance(output, bytearray):
        return bytes(output)
    import logging

    logging.warning(
        "PDF output fallback: encoding with 'latin-1' and 'replace'. Some characters may be replaced."
    )
    return str(output).encode("latin-1", "replace")


def generate_unified_report(
    result: Any,
    config: Any,
    *,
    run_id: str | None = None,
    include_pdf: bool = False,
    spec: TrendRunSpec | None = None,
) -> ReportArtifacts:
    """Produce HTML (and optional PDF) report artifacts for a simulation
    result."""

    backtest = _build_backtest(result)
    exec_summary = _build_exec_summary(result, backtest)
    metrics_df = getattr(result, "metrics", pd.DataFrame())
    metrics_html, metrics_text = _metrics_table_html(metrics_df)
    params = _build_param_summary(config, spec)
    caveats = _build_caveats(result, backtest)
    details_mapping = (
        getattr(result, "details", {})
        if isinstance(getattr(result, "details", None), Mapping)
        else {}
    )
    raw_regime_table = details_mapping.get("performance_by_regime")
    regime_table = (
        raw_regime_table
        if isinstance(raw_regime_table, pd.DataFrame)
        else pd.DataFrame()
    )
    regime_html, regime_text = _format_regime_table(regime_table)
    regime_notes = list(details_mapping.get("regime_notes", []))
    regime_summary = details_mapping.get("regime_summary")
    narrative = _narrative(
        backtest, regime_summary if isinstance(regime_summary, str) else None
    )
    turnover_chart = _turnover_chart(backtest)
    exposure_chart = _exposure_chart(backtest)
    footer = "Past performance does not guarantee future results."
    context: dict[str, Any] = {
        "title": "Vol-Adj Trend Analysis Report",
        "run_id": run_id or getattr(result, "seed", "n/a"),
        "exec_summary": exec_summary,
        "narrative": narrative,
        "metrics_html": metrics_html,
        "metrics_text": metrics_text,
        "regime_html": regime_html,
        "regime_text": regime_text,
        "regime_notes": regime_notes,
        "regime_summary": regime_summary,
        "parameters": params,
        "caveats": caveats
        or [
            "Review risk diagnostics and configuration assumptions before acting on these results."
        ],
        "turnover_chart": turnover_chart,
        "exposure_chart": exposure_chart,
        "footer": footer,
    }
    if spec is None:
        spec = getattr(config, "_trend_run_spec", None)
    if spec is not None:
        context["resolved_spec"] = spec
    html_output = _render_html(context)
    pdf_bytes = None
    if include_pdf:
        pdf_bytes = _render_pdf(context)
    return ReportArtifacts(html=html_output, pdf_bytes=pdf_bytes, context=context)
