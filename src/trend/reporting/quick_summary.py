"""Lightweight reporting over existing run artefacts.

This module inspects the CSV / JSON outputs written by the ``trend`` CLI and
produces a compact HTML report alongside a parameter-sweep heatmap.  It is
designed for automation: a single command ingests artefacts for a run and
emits ``perf/reports/<run-id>.html`` while also persisting
``perf/heatmap_<run-id>.png``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from datetime import UTC, datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd


def _init_matplotlib() -> Any:  # pragma: no cover - thin wrapper
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["savefig.edgecolor"] = "white"
    matplotlib.rcParams["savefig.bbox"] = "tight"
    matplotlib.rcParams["savefig.pad_inches"] = 0.1
    matplotlib.rcParams["savefig.dpi"] = 160
    return plt


plt = _init_matplotlib()


def _coerce_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.copy()
    if isinstance(obj, Mapping):
        return pd.Series(dict(obj), dtype=float)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return pd.Series(list(obj), dtype=float)
    return pd.Series(dtype=float)


def _maybe_datetime_index(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    try:
        converted = pd.to_datetime(series.index)
    except (TypeError, ValueError):
        return series
    return pd.Series(series.values, index=converted)


def _extract_returns(details: Mapping[str, Any]) -> pd.Series:
    candidates = [
        details.get("portfolio_equal_weight_combined"),
        details.get("portfolio_user_weight"),
        details.get("portfolio"),
    ]
    for candidate in candidates:
        series = _coerce_series(candidate)
        if not series.empty:
            return _maybe_datetime_index(series.sort_index())
    return pd.Series(dtype=float)


def _extract_turnover(details: Mapping[str, Any]) -> pd.Series:
    diagnostics = details.get("risk_diagnostics")
    if isinstance(diagnostics, Mapping):
        series = _coerce_series(diagnostics.get("turnover"))
        if not series.empty:
            return _maybe_datetime_index(series.sort_index())
    return pd.Series(dtype=float)


def _extract_parameter_grid(
    details: Mapping[str, Any], metrics: pd.DataFrame
) -> pd.DataFrame:
    payload = details.get("parameter_grid")
    if isinstance(payload, Mapping):
        values = payload.get("values")
        rows = payload.get("rows") or payload.get("index")
        cols = payload.get("cols") or payload.get("columns")
        try:
            grid = pd.DataFrame(values, index=list(rows), columns=list(cols))
        except Exception:
            grid = pd.DataFrame(values)
        if not grid.empty:
            numeric_grid = grid.apply(pd.to_numeric, errors="coerce")
            if numeric_grid.notna().any().any():
                return numeric_grid

    grid_map = details.get("grid")
    if isinstance(grid_map, Mapping):
        try:
            df = pd.DataFrame(grid_map)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            numeric = df.apply(pd.to_numeric, errors="coerce")
            if numeric.notna().any().any():
                return numeric

    numeric_metrics = metrics.select_dtypes(include="number")
    if not numeric_metrics.empty:
        grid = numeric_metrics.T
        return grid.astype(float)
    return pd.DataFrame([[0.0]], index=["metric"], columns=["value"])


def _compute_equity_and_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    if returns.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    ordered = returns.sort_index()
    filled = ordered.fillna(0.0)
    equity = (1.0 + filled).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return equity, drawdown


def _encode_plot(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _equity_drawdown_chart(returns: pd.Series) -> str | None:
    equity, drawdown = _compute_equity_and_drawdown(returns)
    if equity.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.5)
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=1.5)
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _encode_plot(fig)


def _turnover_chart(turnover: pd.Series) -> str | None:
    if turnover.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(turnover.index, turnover.values, color="#ff7f0e", linewidth=1.5)
    ax.set_title("Turnover")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _encode_plot(fig)


def render_parameter_grid_heatmap(
    grid: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Parameter sweep performance",
    cmap: str = "viridis",
) -> Path:
    numeric = grid.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.shape[0] == 0 or numeric.shape[1] == 0:
        numeric = pd.DataFrame([[float("nan")]], index=["metric"], columns=["value"])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(numeric.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(numeric.columns)))
    ax.set_xticklabels(list(map(str, numeric.columns)))
    ax.set_yticks(range(len(numeric.index)))
    ax.set_yticklabels(list(map(str, numeric.index)))
    ax.set_title(title)
    ax.set_xlabel(str(numeric.columns.name or "Parameter B"))
    ax.set_ylabel(str(numeric.index.name or "Parameter A"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _load_metrics(path: Path, run_id: str) -> pd.DataFrame:
    candidate_files = [path / f"metrics_{run_id}.csv", path / "metrics.csv"]
    for candidate in candidate_files:
        if candidate.exists():
            return pd.read_csv(candidate, index_col=0)
    raise FileNotFoundError(f"Metrics file not found for run '{run_id}' in {path}")


def _load_summary(path: Path, run_id: str) -> str | None:
    summary_file = path / f"summary_{run_id}.txt"
    if summary_file.exists():
        return summary_file.read_text(encoding="utf-8")
    legacy_file = path / "summary.txt"
    if legacy_file.exists():
        return legacy_file.read_text(encoding="utf-8")
    return None


def _load_details(path: Path, run_id: str) -> Mapping[str, Any]:
    detail_file = path / f"details_{run_id}.json"
    if detail_file.exists():
        with detail_file.open("r", encoding="utf-8") as fh:
            return cast(Mapping[str, Any], json.load(fh))
    legacy = path / "details.json"
    if legacy.exists():
        with legacy.open("r", encoding="utf-8") as fh:
            return cast(Mapping[str, Any], json.load(fh))
    raise FileNotFoundError(f"Details file not found for run '{run_id}' in {path}")


def _render_html(
    *,
    run_id: str,
    generated_at: datetime,
    config_text: str | None,
    metrics: pd.DataFrame,
    summary_text: str | None,
    equity_chart: str | None,
    turnover_chart: str | None,
    heatmap_rel_path: str,
) -> str:
    numeric_cols = metrics.select_dtypes(include="number").columns
    formatted_metrics = metrics.copy()
    if len(numeric_cols) == len(metrics.columns):
        formatted_metrics = metrics.round(4)
    else:
        formatted_metrics[numeric_cols] = formatted_metrics[numeric_cols].round(4)
    metrics_table = formatted_metrics.to_html(classes="metrics", border=0)
    summary_block = ""
    if summary_text:
        safe = html_escape(summary_text.strip())
        summary_block = f'<pre class="summary">{safe}</pre>'
    config_block = ""
    if config_text:
        safe_cfg = html_escape(config_text.strip())
        config_block = f'<pre class="config">{safe_cfg}</pre>'
    equity_block = (
        f'<img alt="Equity and drawdown" src="data:image/png;base64,{equity_chart}">'
        if equity_chart
        else '<p class="placeholder">No equity data available.</p>'
    )
    turnover_block = (
        f'<img alt="Turnover" src="data:image/png;base64,{turnover_chart}">'
        if turnover_chart
        else '<p class="placeholder">No turnover data available.</p>'
    )
    summary_section = (
        f"<section><h2>Summary</h2>{summary_block}</section>" if summary_block else ""
    )
    config_section = (
        f"<section><h2>Configuration</h2>{config_block}</section>"
        if config_block
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Trend run report – {html_escape(run_id)}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; }}
      h1 {{ margin-bottom: 0.25rem; }}
      h2 {{ margin-top: 1.75rem; }}
      .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
      pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
      table.metrics {{ border-collapse: collapse; width: auto; margin-top: 0.5rem; }}
      table.metrics th, table.metrics td {{
        border: 1px solid #ddd;
        padding: 0.4rem 0.6rem;
        text-align: right;
      }}
      table.metrics th:first-child, table.metrics td:first-child {{ text-align: left; }}
      .placeholder {{ color: #999; font-style: italic; }}
      img {{ max-width: 100%; height: auto; display: block; margin-top: 0.5rem; }}
    </style>
  </head>
  <body>
    <h1>Trend run report</h1>
    <div class=\"meta\">Run ID: {html_escape(run_id)} · Generated {generated_at.isoformat()}</div>
    {summary_section}
    {config_section}
    <section>
      <h2>Key metrics</h2>
      {metrics_table}
    </section>
    <section>
      <h2>Equity &amp; drawdown</h2>
      {equity_block}
    </section>
    <section>
      <h2>Turnover</h2>
      {turnover_block}
    </section>
    <section>
      <h2>Parameter grid</h2>
      <img alt=\"Parameter grid heatmap\" src=\"{html_escape(heatmap_rel_path)}\">
    </section>
  </body>
</html>
"""


def _infer_run_id(artifacts_dir: Path) -> str:
    metrics_files = sorted(artifacts_dir.glob("metrics_*.csv"))
    if metrics_files:
        name = metrics_files[0].stem
        if name.startswith("metrics_"):
            return name.split("metrics_")[1]
    legacy = artifacts_dir / "metrics.csv"
    if legacy.exists():
        return artifacts_dir.name
    raise ValueError(
        f"Unable to infer run identifier from artefacts in {artifacts_dir}"
    )


def build_run_report(
    run_id: str,
    artifacts_dir: Path,
    *,
    config_text: str | None = None,
    output_path: Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    metrics = _load_metrics(artifacts_dir, run_id)
    summary_text = _load_summary(artifacts_dir, run_id)
    details = _load_details(artifacts_dir, run_id)
    returns = _extract_returns(details)
    turnover = _extract_turnover(details)
    grid = _extract_parameter_grid(details, metrics)

    base = base_dir or artifacts_dir.parent
    heatmap_path = base / f"heatmap_{run_id}.png"
    render_parameter_grid_heatmap(grid, heatmap_path)

    destination = output_path or (base / "reports" / f"{run_id}.html")
    heatmap_rel_path = os.path.relpath(heatmap_path, start=destination.parent)

    html_output = _render_html(
        run_id=run_id,
        generated_at=datetime.now(UTC),
        config_text=config_text,
        metrics=metrics,
        summary_text=summary_text,
        equity_chart=_equity_drawdown_chart(returns),
        turnover_chart=_turnover_chart(turnover),
        heatmap_rel_path=heatmap_rel_path,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_output, encoding="utf-8")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trend-quick-report",
        description="Generate a compact HTML report from trend run artefacts.",
    )
    parser.add_argument(
        "--run-id", help="Run identifier (defaults to artefact inference)"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Directory containing metrics_<run-id>.csv and details_<run-id>.json",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("perf"),
        help="Base directory for derived artefacts (default: ./perf)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file used for the run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output HTML path (default: <base-dir>/reports/<run-id>.html)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    artifacts_dir = args.artifacts
    if artifacts_dir is None:
        if args.run_id is None:
            parser.error("--artifacts or --run-id must be supplied")
        artifacts_dir = args.base_dir / args.run_id
    run_id = args.run_id or _infer_run_id(artifacts_dir)
    config_text = None
    if args.config is not None:
        config_text = args.config.read_text(encoding="utf-8")

    try:
        build_run_report(
            run_id,
            artifacts_dir,
            config_text=config_text,
            output_path=args.output,
            base_dir=args.base_dir,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        parser.exit(2, f"Error: {exc}\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
