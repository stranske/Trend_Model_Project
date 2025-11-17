"""Utilities for writing per-run manifests and HTML receipts."""

from __future__ import annotations

import datetime as _dt
import html
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from trend_analysis.util.hash import normalise_for_json, sha256_config, sha256_file

_METRIC_FIELDS = (
    "cagr",
    "vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "information_ratio",
)


def _git_hash() -> str:
    """Return the current git commit hash when available."""

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], encoding="utf-8", shell=False
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _serialise_stats(stats: Any) -> dict[str, float]:
    """Convert stats-like objects to a simple mapping of floats."""

    if stats is None:
        return {}
    values: dict[str, float] = {}
    for field in _METRIC_FIELDS:
        value = None
        if isinstance(stats, Mapping):
            value = stats.get(field)
        else:
            value = getattr(stats, field, None)
        if value is None:
            continue
        try:
            values[field] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _coerce_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    try:
        return pd.DataFrame(frame)
    except Exception:
        return pd.DataFrame()


def _data_window(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(df.shape[0])}
    date_series = None
    for col in df.columns:
        if str(col).lower() == "date":
            date_series = pd.to_datetime(df[col], errors="coerce")
            break
    if date_series is None:
        index = df.index
        if isinstance(index, pd.PeriodIndex):
            date_series = index.to_timestamp()
        elif isinstance(index, pd.DatetimeIndex):
            date_series = index
    if date_series is not None:
        valid = date_series.dropna()
        if not valid.empty:
            summary["start"] = valid.min().isoformat()
            summary["end"] = valid.max().isoformat()
    instrument_cols = [c for c in df.columns if str(c).lower() != "date"]
    summary["instrument_count"] = len(instrument_cols)
    return summary


def _summarise_metrics(df: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    if df.empty:
        return summary
    numeric = df.select_dtypes(include=["number"]).copy()
    for field in _METRIC_FIELDS:
        if field not in numeric:
            continue
        series = pd.to_numeric(numeric[field], errors="coerce").dropna()
        if series.empty:
            continue
        summary[f"avg_{field}"] = float(series.mean())
        summary[f"max_{field}"] = float(series.max())
        summary[f"min_{field}"] = float(series.min())
    return summary


def _render_html(
    *,
    run_id: str,
    created: _dt.datetime,
    manifest: Mapping[str, Any],
    summary_text: str,
) -> str:
    metrics = manifest.get("metrics", {})
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(f'{v:.4f}' if isinstance(v, (int, float)) else str(v))}</td></tr>"
        for k, v in metrics.items()
    )
    artifacts = manifest.get("artifacts", [])
    artifact_rows = "".join(
        f"<li><a href=\"{html.escape(str(item.get('name', '')))}\">{html.escape(str(item.get('name', '')))}</a>"
        f" ({item.get('size', 0)} bytes)</li>"
        for item in artifacts
    ) or "<li>No exported artifacts were detected.</li>"
    data_window = manifest.get("data_window", {})
    date_range = " / ".join(
        filter(None, [str(data_window.get("start")), str(data_window.get("end"))])
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Trend Analysis Run {html.escape(run_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }}
    header {{ margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background-color: #f5f5f5; }}
    ul {{ padding-left: 1.2rem; }}
    pre {{ background: #f8f8f8; padding: 1rem; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>Trend run receipt</h1>
    <p><strong>Run ID:</strong> {html.escape(run_id)}<br>
       <strong>Created:</strong> {html.escape(created.isoformat())}<br>
       <strong>Git hash:</strong> {html.escape(str(manifest.get('git_hash', 'unknown')))}</p>
    <p><strong>Data window:</strong> {html.escape(date_range or 'unknown')} ·
       <strong>Instruments:</strong> {html.escape(str(data_window.get('instrument_count', 'n/a')))}</p>
  </header>
  <section>
    <h2>Key metrics</h2>
    <table>
      <tbody>
        {metric_rows or '<tr><td colspan=\"2\">No metrics recorded.</td></tr>'}
      </tbody>
    </table>
  </section>
  <section>
    <h2>Artifacts</h2>
    <ul>
      {artifact_rows}
    </ul>
  </section>
  <section>
    <h2>Summary</h2>
    <pre>{html.escape(summary_text or 'No summary text provided.')}</pre>
  </section>
</body>
</html>"""


def write_run_artifacts(
    *,
    output_dir: Path,
    run_id: str,
    config: Any,
    config_path: str,
    input_path: Path,
    data_frame: Any,
    metrics_frame: Any,
    run_details: Mapping[str, Any] | None,
    exported_files: Sequence[Path],
    summary_text: str,
) -> Path:
    """Copy exported files into a timestamped directory with manifest + HTML."""

    created = _dt.datetime.now(_dt.timezone.utc)
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = run_id[:8] or "run"
    run_root = base_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / f"{created.strftime('%Y%m%d_%H%M%S')}_{run_prefix}"
    run_dir.mkdir(exist_ok=True)

    df = _coerce_frame(data_frame)
    metrics_df = _coerce_frame(metrics_frame)
    details = dict(run_details or {})

    copied: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in exported_files:
        src = Path(path)
        if src in seen:
            continue
        seen.add(src)
        if not src.exists():
            continue
        dest = run_dir / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(
            {
                "name": dest.name,
                "source": str(src),
                "size": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        )

    selected = details.get("selected_funds")
    if isinstance(selected, Sequence) and not isinstance(
        selected, (str, bytes, bytearray)
    ):
        selected_list = list(selected)
    elif selected is None:
        selected_list = []
    else:
        selected_list = [selected]

    manifest: dict[str, Any] = {
        "schema_version": "trend.run_artifacts/1",
        "run_id": run_id,
        "created": created.isoformat().replace("+00:00", "Z"),
        "config_path": config_path,
        "input_path": str(input_path),
        "config_sha256": sha256_config(config),
        "git_hash": _git_hash(),
        "data_window": _data_window(df),
        "metrics": _serialise_stats(details.get("out_ew_stats")) or _summarise_metrics(metrics_df),
        "metrics_overview": _summarise_metrics(metrics_df),
        "selected_funds": selected_list,
        "artifacts": copied,
    }
    if config is not None:
        manifest["config_snapshot"] = normalise_for_json(config)
    if input_path.exists():
        try:
            manifest["input_sha256"] = sha256_file(input_path)
        except OSError:
            pass
    manifest["summary_text"] = summary_text
    manifest["html_report"] = "report.html"
    manifest["run_directory"] = str(run_dir)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(normalise_for_json(manifest), indent=2), encoding="utf-8")

    html_path = run_dir / "report.html"
    html_path.write_text(
        _render_html(run_id=run_id, created=created, manifest=manifest, summary_text=summary_text),
        encoding="utf-8",
    )

    return run_dir


__all__ = ["write_run_artifacts"]
