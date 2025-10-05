"""Market regime classification and performance aggregation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metrics import annual_return, max_drawdown, sharpe_ratio
from .perf.rolling_cache import compute_dataset_hash, get_cache


@dataclass(frozen=True)
class RegimeSettings:
    """Normalised configuration controlling regime detection."""

    enabled: bool = False
    proxy: str | None = None
    method: str = "rolling_return"
    lookback: int = 126
    smoothing: int = 3
    threshold: float = 0.0
    neutral_band: float = 0.001
    min_obs: int = 6
    risk_on_label: str = "Risk-On"
    risk_off_label: str = "Risk-Off"
    default_label: str = "Risk-On"
    cache: bool = True
    annualise_volatility: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return max(default, minimum)
    if num < minimum:
        return minimum
    return num


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalise_settings(cfg: Mapping[str, Any] | None) -> RegimeSettings:
    """Return :class:`RegimeSettings` populated from a mapping."""

    if cfg is None:
        return RegimeSettings()

    enabled = bool(cfg.get("enabled", False))
    proxy = cfg.get("proxy")
    if proxy is not None:
        proxy = str(proxy).strip() or None

    method_raw = (
        str(cfg.get("method", "rolling_return") or "rolling_return").strip().lower()
    )
    method_lookup = {
        "rolling_return": "rolling_return",
        "rolling": "rolling_return",
        "return": "rolling_return",
        "volatility": "volatility",
        "vol": "volatility",
        "std": "volatility",
    }
    method = method_lookup.get(method_raw, "rolling_return")

    lookback = _coerce_positive_int(cfg.get("lookback"), 126)
    smoothing = _coerce_positive_int(cfg.get("smoothing"), 3)
    threshold = _coerce_float(cfg.get("threshold"), 0.0)
    neutral_band = abs(_coerce_float(cfg.get("neutral_band"), 0.001))
    min_obs = _coerce_positive_int(cfg.get("min_observations"), 6, minimum=1)
    cache = bool(cfg.get("cache", True))
    annualise_volatility = bool(cfg.get("annualise_volatility", True))

    risk_on_label = str(cfg.get("risk_on_label", "Risk-On") or "Risk-On").strip()
    risk_off_label = str(cfg.get("risk_off_label", "Risk-Off") or "Risk-Off").strip()
    default_label = str(
        cfg.get("default_label", risk_on_label) or risk_on_label
    ).strip()
    if not risk_on_label:
        risk_on_label = "Risk-On"
    if not risk_off_label:
        risk_off_label = "Risk-Off"
    if not default_label:
        default_label = risk_on_label

    return RegimeSettings(
        enabled=enabled,
        proxy=proxy,
        method=method,
        lookback=lookback,
        smoothing=smoothing,
        threshold=threshold,
        neutral_band=neutral_band,
        min_obs=min_obs,
        risk_on_label=risk_on_label,
        risk_off_label=risk_off_label,
        default_label=default_label,
        cache=cache,
        annualise_volatility=annualise_volatility,
    )


def _rolling_return_signal(
    series: pd.Series, *, window: int, smoothing: int
) -> pd.Series:
    """Return rolling compounded returns smoothed by ``smoothing`` window."""

    if window <= 0:
        raise ValueError("window must be positive")
    compounded = (1.0 + series).rolling(window).apply(np.prod, raw=True) - 1.0
    if smoothing > 1:
        compounded = compounded.rolling(smoothing).mean()
    return compounded


def _rolling_volatility_signal(
    series: pd.Series,
    *,
    window: int,
    smoothing: int,
    periods_per_year: float | None,
    annualise: bool,
) -> pd.Series:
    """
    Return rolling realised volatility optionally annualised.

    Note:
        The rolling standard deviation is calculated with ``ddof=0`` (population standard deviation),
        which differs from the pandas default of ``ddof=1`` (sample standard deviation).
        This choice affects the volatility calculation and may not be obvious to users.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    vol = series.rolling(window).std(ddof=0)
    if smoothing > 1:
        vol = vol.rolling(smoothing).mean()
    if annualise and periods_per_year and periods_per_year > 0:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def _default_periods_per_year(freq: str) -> float:
    """Best-effort mapping from frequency code to periods-per-year."""

    freq_upper = (freq or "").upper()
    if freq_upper.startswith(("A", "Y")):
        return 1.0
    if freq_upper.startswith("Q"):
        return 4.0
    if freq_upper.startswith("M"):
        return 12.0
    if freq_upper.startswith("W"):
        return 52.0
    if freq_upper.startswith(("B", "D")):
        return 252.0
    return 252.0


def _compute_regime_series(
    proxy: pd.Series,
    settings: RegimeSettings,
    *,
    freq: str,
    periods_per_year: float | None,
) -> pd.Series:
    """Classify ``proxy`` observations into regimes using ``settings``."""

    if proxy.empty:
        return pd.Series(dtype="string")

    clean = proxy.astype(float).dropna()
    if clean.empty:
        return pd.Series(dtype="string")

    window = max(int(settings.lookback), 1)
    smoothing = max(int(settings.smoothing), 1)

    periods = None
    if periods_per_year is not None and periods_per_year > 0:
        periods = float(periods_per_year)
    elif settings.method == "volatility":
        periods = _default_periods_per_year(freq)

    if settings.method == "volatility":
        signal = _rolling_volatility_signal(
            clean,
            window=window,
            smoothing=smoothing,
            periods_per_year=periods,
            annualise=settings.annualise_volatility,
        )
        signal = settings.threshold - signal
    else:
        signal = _rolling_return_signal(clean, window=window, smoothing=smoothing)
        signal = signal - settings.threshold

    labels = pd.Series(settings.default_label, index=clean.index, dtype="string")
    upper = settings.neutral_band
    lower = -settings.neutral_band
    if upper <= 0:
        labels.loc[signal >= 0] = settings.risk_on_label
        labels.loc[signal < 0] = settings.risk_off_label
    else:
        labels.loc[signal >= upper] = settings.risk_on_label
        labels.loc[signal <= lower] = settings.risk_off_label
    labels = labels.ffill().bfill()

    if len(labels) < window:
        # Not enough history to form a stable view
        return labels

    if not settings.cache:
        return labels

    dataset_hash = compute_dataset_hash([clean])
    cache = get_cache()
    method_tag = (
        f"regime_{settings.method}_thr{settings.threshold:.6f}_"
        f"smooth{smoothing}_band{settings.neutral_band:.6f}"
    )
    if settings.method == "volatility":
        method_tag = f"{method_tag}_annual{int(settings.annualise_volatility)}"
        if periods:
            method_tag = f"{method_tag}_ppy{periods:.6f}"

    def _compute() -> pd.Series:
        return labels

    return cache.get_or_compute(dataset_hash, window, freq, method_tag, _compute)


def compute_regimes(
    proxy: pd.Series,
    settings: RegimeSettings,
    *,
    freq: str,
    periods_per_year: float | None = None,
) -> pd.Series:
    """Return per-period regime labels for ``proxy`` respecting ``settings``."""

    if not settings.enabled:
        return pd.Series(dtype="string")
    return _compute_regime_series(
        proxy, settings, freq=freq, periods_per_year=periods_per_year
    )


def _format_hit_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    total = series.count()
    if total == 0:
        return float("nan")
    return float((series > 0).sum() / total)


def _summarise_regime_outcome(
    settings: RegimeSettings,
    risk_on: pd.Series | None,
    risk_off: pd.Series | None,
) -> str | None:
    """Return a human-readable comparison between risk-on and risk-off results."""

    def _extract_cagr(series: pd.Series | None) -> float | None:
        if not isinstance(series, pd.Series):
            return None
        value = series.get("CAGR")
        if pd.isna(value):
            return None
        return float(value)

    cagr_on = _extract_cagr(risk_on)
    cagr_off = _extract_cagr(risk_off)
    if cagr_on is None and cagr_off is None:
        return None

    on_label = settings.risk_on_label
    off_label = settings.risk_off_label

    if cagr_on is None:
        return (
            f"{off_label} windows delivered {cagr_off:.1%} CAGR; insufficient data "
            f"for {on_label.lower()} periods."
        )
    if cagr_off is None:
        return (
            f"{on_label} windows delivered {cagr_on:.1%} CAGR; insufficient data "
            f"for {off_label.lower()} periods."
        )

    tolerance = 0.001  # ≈10 bps difference treated as "similar"
    delta = cagr_on - cagr_off
    if abs(delta) <= tolerance:
        return (
            f"Performance was similar across regimes ({on_label} ~{cagr_on:.1%} "
            f"versus {off_label.lower()} {cagr_off:.1%})."
        )
    if delta > 0:
        return (
            f"{on_label} windows delivered {cagr_on:.1%} CAGR, outpacing "
            f"{off_label.lower()} periods at {cagr_off:.1%}."
        )
    return (
        f"{off_label} windows outperformed {on_label.lower()} stretches "
        f"({cagr_off:.1%} vs {cagr_on:.1%} CAGR)."
    )


def aggregate_performance_by_regime(
    returns_map: Mapping[str, pd.Series],
    risk_free: pd.Series | float,
    regimes: pd.Series,
    settings: RegimeSettings,
    *,
    periods_per_year: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate performance metrics conditioned on ``regimes``."""

    if not settings.enabled or not returns_map:
        return pd.DataFrame(), []

    if regimes.empty:
        return pd.DataFrame(), [
            "Regime labels were unavailable for the analysis window."
        ]

    # Ensure string dtype for comparisons and align to returns index
    regimes = regimes.astype("string")

    portfolios = {
        str(name): series.astype(float) for name, series in returns_map.items()
    }
    all_index = pd.Index([])
    for series in portfolios.values():
        all_index = all_index.union(series.index)

    regimes = regimes.reindex(all_index).ffill().bfill()

    columns = []
    for portfolio in portfolios:
        for regime in (settings.risk_on_label, settings.risk_off_label, "All"):
            columns.append((portfolio, regime))

    idx = ["CAGR", "Sharpe", "Max Drawdown", "Hit Rate", "Observations"]
    table = pd.DataFrame(
        np.nan,
        index=idx,
        columns=pd.MultiIndex.from_tuples(columns, names=["portfolio", "regime"]),
        dtype=float,
    )

    notes: list[str] = []

    for portfolio, series in portfolios.items():
        aligned = series.reindex(all_index).astype(float)
        rf_aligned: pd.Series | float
        if isinstance(risk_free, pd.Series):
            rf_aligned = risk_free.reindex(all_index).astype(float)
        else:
            rf_aligned = float(risk_free)

        for regime in (settings.risk_on_label, settings.risk_off_label):
            mask = regimes == regime
            subset = aligned[mask]
            if isinstance(rf_aligned, pd.Series):
                rf_subset = rf_aligned[mask]
            else:
                rf_subset = rf_aligned
            obs = int(subset.count())
            table.loc["Observations", (portfolio, regime)] = float(obs)
            if obs < settings.min_obs:
                notes.append(
                    f"{regime} regime has fewer than {settings.min_obs} observations; "
                    "metrics shown as N/A."
                )
                continue
            table.loc["CAGR", (portfolio, regime)] = float(
                annual_return(subset.dropna(), periods_per_year=int(periods_per_year))
            )
            table.loc["Sharpe", (portfolio, regime)] = float(
                sharpe_ratio(
                    subset.dropna(), rf_subset, periods_per_year=int(periods_per_year)
                )
            )
            table.loc["Max Drawdown", (portfolio, regime)] = float(
                max_drawdown(subset.dropna())
            )
            table.loc["Hit Rate", (portfolio, regime)] = _format_hit_rate(
                subset.dropna()
            )

        # All periods aggregate
        subset_all = aligned.dropna()
        if isinstance(rf_aligned, pd.Series):
            rf_all = rf_aligned.reindex(subset_all.index).dropna()
        else:
            rf_all = rf_aligned
        obs_all = int(subset_all.count())
        table.loc["Observations", (portfolio, "All")] = float(obs_all)
        if obs_all >= settings.min_obs:
            table.loc["CAGR", (portfolio, "All")] = float(
                annual_return(subset_all, periods_per_year=int(periods_per_year))
            )
            table.loc["Sharpe", (portfolio, "All")] = float(
                sharpe_ratio(subset_all, rf_all, periods_per_year=int(periods_per_year))
            )
            table.loc["Max Drawdown", (portfolio, "All")] = float(
                max_drawdown(subset_all)
            )
            table.loc["Hit Rate", (portfolio, "All")] = _format_hit_rate(subset_all)
        else:
            notes.append(
                f"All-period aggregate for {portfolio} has fewer than "
                f"{settings.min_obs} observations; metrics shown as N/A."
            )

    if notes:
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for note in notes:
            if note not in seen:
                deduped.append(note)
                seen.add(note)
        notes = deduped

    return table, notes


def build_regime_payload(
    *,
    data: pd.DataFrame,
    out_index: pd.Index,
    returns_map: Mapping[str, pd.Series],
    risk_free: pd.Series | float,
    config: Mapping[str, Any] | None,
    freq_code: str,
    periods_per_year: float,
) -> dict[str, Any]:
    """Compute regime labels and aggregates for the provided inputs."""

    settings = normalise_settings(config)
    payload: dict[str, Any] = {
        "settings": settings.to_dict(),
        "labels": pd.Series(dtype="string"),
        "out_labels": pd.Series(dtype="string"),
        "table": pd.DataFrame(),
        "notes": [],
        "summary": None,
    }

    if not settings.enabled:
        payload["notes"] = ["Regime analysis disabled in configuration."]
        return payload

    if not settings.proxy:
        payload["notes"] = [
            "Regime proxy column not specified; skipping regime analysis."
        ]
        return payload

    if settings.proxy not in data.columns:
        payload["notes"] = [
            f"Proxy column '{settings.proxy}' not found in input data; regime analysis skipped.",
        ]
        return payload

    proxy_series = data.set_index("Date")[settings.proxy].astype(float)
    regimes = compute_regimes(
        proxy_series,
        settings,
        freq=freq_code,
        periods_per_year=periods_per_year,
    )
    if regimes.empty:
        payload["notes"] = [
            "Market proxy series did not produce regime labels for the requested window.",
        ]
        return payload

    payload["labels"] = regimes

    out_labels = regimes.reindex(out_index).ffill().bfill()
    if out_labels.isna().any():
        out_labels = out_labels.fillna(settings.default_label)
        payload.setdefault("notes", []).append(
            "Regime labels contained gaps; forward/backward fill applied for reporting."
        )
    payload["out_labels"] = out_labels

    table, notes = aggregate_performance_by_regime(
        returns_map,
        risk_free,
        out_labels,
        settings,
        periods_per_year=periods_per_year,
    )
    payload["table"] = table

    missing = out_index.difference(out_labels.index)
    if len(missing) > 0:
        payload.setdefault("notes", []).append(
            "Proxy series missing for portions of the out-of-sample window; regime labels "
            "may be incomplete."
        )

    payload.setdefault("notes", []).extend(notes)

    if not table.empty:
        # Build a compact summary sentence using the first portfolio columns
        user_cols = [col for col in table.columns if col[1] != "All"]
        if user_cols:
            first = user_cols[0]
            summary_text = _summarise_regime_outcome(
                settings,
                table.get((first[0], settings.risk_on_label)),
                table.get((first[0], settings.risk_off_label)),
            )
            if summary_text:
                payload["summary"] = summary_text

    if payload["summary"] is None and notes:
        payload["summary"] = notes[0]

    if payload.get("notes"):
        # Deduplicate notes while preserving insertion order
        seen_notes: set[str] = set()
        ordered_notes: list[str] = []
        for note in payload["notes"]:
            if note not in seen_notes:
                ordered_notes.append(note)
                seen_notes.add(note)
        payload["notes"] = ordered_notes

    return payload


__all__ = [
    "RegimeSettings",
    "aggregate_performance_by_regime",
    "build_regime_payload",
    "compute_regimes",
    "normalise_settings",
]
