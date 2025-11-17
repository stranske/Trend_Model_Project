"""Helpers for loading TOML configs into strongly typed run specifications."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

try:  # Python 3.11+
    import tomllib  # pragma: no cover - exercised in production
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[no-redef]

from trend_analysis.backtesting import CostModel
from trend_analysis.config import load_config
from trend_analysis.signals import TrendSpec

__all__ = [
    "SampleWindow",
    "BacktestSpec",
    "TrendRunSpec",
    "load_run_spec_from_file",
    "load_run_spec_from_mapping",
    "ensure_run_spec",
]


@dataclass(frozen=True, slots=True)
class SampleWindow:
    """Resolved in/out of sample window boundaries."""

    in_start: str
    in_end: str
    out_start: str
    out_end: str


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    """Configuration required to execute the analysis and backtest."""

    window: SampleWindow
    selection_mode: str
    random_n: int
    rebalance_calendar: str | None
    transaction_cost_bps: float
    cost_model: CostModel
    max_turnover: float | None
    rank: Mapping[str, Any]
    selector: Mapping[str, Any]
    weighting: Mapping[str, Any]
    weighting_scheme: str | None
    custom_weights: Mapping[str, float] | None
    manual_list: tuple[str, ...]
    indices_list: tuple[str, ...]
    benchmarks: Mapping[str, str]
    missing: Mapping[str, Any]
    target_vol: float | None
    floor_vol: float | None
    warmup_periods: int
    monthly_cost: float
    previous_weights: Mapping[str, float] | None
    regime: Mapping[str, Any]
    metrics: tuple[str, ...]
    seed: int
    jobs: int | None
    checkpoint_dir: Path | None
    export_directory: Path | None
    export_formats: tuple[str, ...]
    output_path: Path | None
    output_format: str | None
    multi_period: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrendRunSpec:
    """Bundle of resolved ``TrendSpec`` and ``BacktestSpec``."""

    trend: TrendSpec
    backtest: BacktestSpec
    config: Any


def _as_mapping(section: Any) -> Mapping[str, Any]:
    if isinstance(section, Mapping):
        return section
    getter = getattr(section, "model_dump", None)
    if callable(getter):
        dumped = getter()
        if isinstance(dumped, Mapping):
            return cast(Mapping[str, Any], dumped)
    attrs = getattr(section, "__dict__", None)
    if isinstance(attrs, Mapping):
        return cast(Mapping[str, Any], attrs)
    return cast(Mapping[str, Any], {})


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_section(cfg: Any, key: str) -> Mapping[str, Any]:
    section = _cfg_value(cfg, key, {})
    return _as_mapping(section)


def _section_get(section: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return section.get(key, default)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_tuple(
    sequence: Sequence[Any] | None, *, coerce: Callable[[Any], Any] | None = None
) -> tuple[Any, ...]:
    if sequence is None:
        return ()
    if coerce is None:
        return tuple(sequence) if not isinstance(sequence, tuple) else sequence
    return tuple(coerce(item) for item in sequence)


def _maybe_path(value: Any, *, base_path: Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_path is not None:
        path = (base_path / path).resolve()
    return path


def _build_trend_spec(cfg: Any) -> TrendSpec:
    signals = _cfg_section(cfg, "signals")
    vol_adjust = bool(_section_get(signals, "vol_adjust", False))
    vol_target = _coerce_float(_section_get(signals, "vol_target"))
    if not vol_adjust:
        vol_target = None
    min_periods_raw = _section_get(signals, "min_periods")
    min_periods = _coerce_int(min_periods_raw, default=0) or None
    if min_periods is not None and min_periods <= 0:
        min_periods = None
    return TrendSpec(
        kind="tsmom",
        window=max(1, _coerce_int(_section_get(signals, "window", 63), default=63)),
        min_periods=min_periods,
        lag=max(1, _coerce_int(_section_get(signals, "lag", 1), default=1)),
        vol_adjust=vol_adjust,
        vol_target=vol_target,
        zscore=bool(_section_get(signals, "zscore", False)),
    )


def _build_backtest_spec(cfg: Any, *, base_path: Path | None) -> BacktestSpec:
    sample = _cfg_section(cfg, "sample_split")
    portfolio = _cfg_section(cfg, "portfolio")
    vol_adjust = _cfg_section(cfg, "vol_adjust")
    preprocessing = _cfg_section(cfg, "preprocessing")
    missing = _cfg_section(preprocessing, "missing_data") if preprocessing else {}
    run_cfg = _cfg_section(cfg, "run")
    export_cfg = _cfg_section(cfg, "export")
    output_cfg = _cfg_section(cfg, "output")
    window = SampleWindow(
        in_start=str(_section_get(sample, "in_start", "")),
        in_end=str(_section_get(sample, "in_end", "")),
        out_start=str(_section_get(sample, "out_start", "")),
        out_end=str(_section_get(sample, "out_end", "")),
    )
    rank_cfg = _as_mapping(_section_get(portfolio, "rank", {}))
    selector_cfg = _as_mapping(_section_get(portfolio, "selector", {}))
    weighting_cfg = _as_mapping(_section_get(portfolio, "weighting", {}))
    metrics = _cfg_section(cfg, "metrics")
    multi_period = _cfg_section(cfg, "multi_period")
    benchmarks = _cfg_section(cfg, "benchmarks")
    manual = _section_get(portfolio, "manual_list")
    indices = _section_get(portfolio, "indices_list")
    export_dir = _maybe_path(_section_get(export_cfg, "directory"), base_path=base_path)
    output_path = _maybe_path(_section_get(output_cfg, "path"), base_path=base_path)
    checkpoint = _maybe_path(
        _section_get(run_cfg, "checkpoint_dir"), base_path=base_path
    )
    cost_cfg = _cfg_section(portfolio, "cost_model")
    slippage = float(_coerce_float(cost_cfg.get("slippage_bps"), 0.0) or 0.0)
    override_bps = _coerce_float(cost_cfg.get("bps_per_trade"))
    effective_bps = (
        float(override_bps)
        if override_bps is not None
        else float(transaction_cost_bps)
    )
    cost_model = CostModel(bps_per_trade=effective_bps, slippage_bps=slippage)

    return BacktestSpec(
        window=window,
        selection_mode=str(_section_get(portfolio, "selection_mode", "all")),
        random_n=_coerce_int(_section_get(portfolio, "random_n", 0), default=0),
        rebalance_calendar=_section_get(portfolio, "rebalance_calendar"),
        transaction_cost_bps=float(
            _coerce_float(_section_get(portfolio, "transaction_cost_bps", 0.0), 0.0)
            or 0.0
        ),
        cost_model=cost_model,
        max_turnover=_coerce_float(_section_get(portfolio, "max_turnover")),
        rank=rank_cfg,
        selector=selector_cfg,
        weighting=weighting_cfg,
        weighting_scheme=_section_get(portfolio, "weighting_scheme"),
        custom_weights=_section_get(portfolio, "custom_weights"),
        manual_list=_as_tuple(manual, coerce=str),
        indices_list=_as_tuple(indices, coerce=str),
        benchmarks=benchmarks,
        missing=missing,
        target_vol=_coerce_float(_section_get(vol_adjust, "target_vol")),
        floor_vol=_coerce_float(_section_get(vol_adjust, "floor_vol")),
        warmup_periods=_coerce_int(
            _section_get(vol_adjust, "warmup_periods", 0), default=0
        ),
        monthly_cost=float(
            _coerce_float(_section_get(run_cfg, "monthly_cost", 0.0), 0.0) or 0.0
        ),
        previous_weights=_section_get(portfolio, "previous_weights"),
        regime=_cfg_section(cfg, "regime"),
        metrics=_as_tuple(_section_get(metrics, "registry", ())),
        seed=_coerce_int(
            _cfg_value(cfg, "seed", _section_get(run_cfg, "seed", 42)), default=42
        ),
        jobs=_coerce_int(_section_get(run_cfg, "jobs", None), default=0) or None,
        checkpoint_dir=checkpoint,
        export_directory=export_dir,
        export_formats=_as_tuple(_section_get(export_cfg, "formats", ())),
        output_path=output_path,
        output_format=_section_get(output_cfg, "format"),
        multi_period=multi_period,
    )


@contextlib.contextmanager
def _temporary_cwd(directory: Path | None) -> Iterator[None]:
    if directory is None:
        yield
        return
    previous = Path.cwd()
    try:
        os.chdir(directory)
    except FileNotFoundError:
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def load_run_spec_from_mapping(
    payload: Mapping[str, Any], *, base_path: Path | None = None
) -> TrendRunSpec:
    """Validate ``payload`` and return resolved run specifications."""

    with _temporary_cwd(base_path):
        cfg = load_config(payload)
    trend_spec = _build_trend_spec(cfg)
    backtest_spec = _build_backtest_spec(cfg, base_path=base_path)
    return TrendRunSpec(trend=trend_spec, backtest=backtest_spec, config=cfg)


def load_run_spec_from_file(path: Path) -> TrendRunSpec:
    """Load a TOML configuration file into resolved run specifications."""

    cfg_path = path.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, Mapping):
        raise TypeError("Configuration files must contain a top-level table")
    return load_run_spec_from_mapping(data, base_path=cfg_path.parent)


def ensure_run_spec(cfg: Any, *, base_path: Path | None = None) -> TrendRunSpec | None:
    """Attach resolved spec dataclasses to ``cfg`` when possible."""

    try:
        trend_spec = _build_trend_spec(cfg)
        backtest_spec = _build_backtest_spec(cfg, base_path=base_path)
    except Exception:
        return None
    spec = TrendRunSpec(trend=trend_spec, backtest=backtest_spec, config=cfg)
    for attr, value in {
        "_trend_run_spec": spec,
        "trend_spec": trend_spec,
        "backtest_spec": backtest_spec,
    }.items():
        try:
            setattr(cfg, attr, value)
        except Exception:
            try:
                # Fallback: forcibly set attribute even if cfg is a frozen dataclass or has custom __setattr__.
                # This bypasses attribute access controls and should only be used when normal setattr fails.
                object.__setattr__(cfg, attr, value)
            except Exception:
                continue
    return spec
