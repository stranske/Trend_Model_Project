"""Utilities for writing per-run manifests and HTML receipts."""

from __future__ import annotations

import datetime as _dt
import html
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from trend_analysis.identity import IdentityMap
from trend_analysis.util.git import git_hash
from trend_analysis.util.hash import normalise_for_json, sha256_config, sha256_file

_METRIC_FIELDS = (
    "cagr",
    "vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "information_ratio",
)


_git_hash = git_hash


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


def _metadata_attr(df: pd.DataFrame, name: str, default: Any = None) -> Any:
    market_data = df.attrs.get("market_data", {})
    metadata = market_data.get("metadata") if isinstance(market_data, Mapping) else None
    if isinstance(metadata, Mapping) and name in metadata:
        return metadata[name]
    if metadata is not None and hasattr(metadata, name):
        return getattr(metadata, name)
    if isinstance(market_data, Mapping) and name in market_data:
        return market_data[name]
    attr_name = f"market_data_{name}"
    return df.attrs.get(attr_name, default)


def _serialise_fill_details(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        raw = value.model_dump()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {"method": str(value)}
    return {str(k): v for k, v in raw.items() if v is not None}


def _coerce_label_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(label): details for label, details in value.items()}
    return {}


def _coerce_label_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(label) for label in value]
    return []


def _data_reality(df: pd.DataFrame) -> dict[str, Any]:
    instrument_cols = [c for c in df.columns if str(c).lower() != "date"]
    instrument_lookup = {str(column): column for column in instrument_cols}
    filled_raw = _coerce_label_mapping(_metadata_attr(df, "missing_policy_filled", {}) or {})
    dropped_labels = _coerce_label_list(_metadata_attr(df, "missing_policy_dropped", []) or [])
    dropped_set = set(dropped_labels)
    filled = {str(label): _serialise_fill_details(details) for label, details in filled_raw.items()}

    instruments: list[dict[str, Any]] = []
    labels_in_order = list(instrument_lookup.keys())
    for dropped_label in dropped_labels:
        if dropped_label not in instrument_lookup:
            labels_in_order.append(dropped_label)

    for label in labels_in_order:
        column = instrument_lookup.get(label)
        fill_details = filled.get(label)
        if fill_details is not None:
            missing_count = int(fill_details.get("count") or 0)
            disposition = "filled"
            reason = fill_details.get("method") or "missing_policy_filled"
        elif label in dropped_set:
            missing_count = int(df[column].isna().sum()) if column is not None else 0
            disposition = "dropped"
            reason = "missing_policy_dropped"
        else:
            missing_count = int(df[column].isna().sum()) if column is not None else 0
            disposition = "kept"
            reason = "complete" if missing_count == 0 else "missing_values_present"
        instruments.append(
            {
                "label": label,
                "missing_count": missing_count,
                "disposition": disposition,
                "reason": str(reason),
            }
        )

    dropped = [
        {
            "label": label,
            "disposition": "dropped",
            "reason": "missing_policy_dropped",
        }
        for label in dropped_labels
    ]

    return {
        "policy": _metadata_attr(df, "missing_policy"),
        "policy_limit": _metadata_attr(df, "missing_policy_limit"),
        "policy_summary": _metadata_attr(df, "missing_policy_summary"),
        "frequency_missing_periods": _metadata_attr(df, "frequency_missing_periods", 0),
        "frequency_max_gap_periods": _metadata_attr(df, "frequency_max_gap_periods", 0),
        "instruments": instruments,
        "filled": [
            {
                "label": label,
                "disposition": "filled",
                "reason": str(details.get("method") or "missing_policy_filled"),
                **details,
            }
            for label, details in filled.items()
        ],
        "dropped": dropped,
        "unknown": [],
    }


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
    metric_rows_parts: list[str] = []
    for key, value in metrics.items():
        display = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        metric_rows_parts.append(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(display)}</td></tr>"
        )
    metric_rows = "".join(metric_rows_parts)
    artifacts = manifest.get("artifacts", [])
    artifact_rows = (
        "".join(
            (
                f"<li><a href='{html.escape(str(item.get('name', '')))}'>"
                f"{html.escape(str(item.get('name', '')))}</a>"
                f" ({item.get('size', 0)} bytes)</li>"
            )
            for item in artifacts
        )
        or "<li>No exported artifacts were detected.</li>"
    )
    data_window = manifest.get("data_window", {})
    date_range = " / ".join(
        filter(None, [str(data_window.get("start")), str(data_window.get("end"))])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
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
       <strong>Git hash:</strong> {html.escape(str(manifest.get("git_hash", "unknown")))}</p>
    <p><strong>Data window:</strong> {html.escape(date_range or "unknown")} ·
       <strong>Instruments:</strong> {html.escape(str(data_window.get("instrument_count", "n/a")))}</p>
  </header>
  <section>
    <h2>Key metrics</h2>
    <table>
      <tbody>
        {metric_rows or '<tr><td colspan="2">No metrics recorded.</td></tr>'}
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
    <pre>{html.escape(summary_text or "No summary text provided.")}</pre>
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
    identity_map: IdentityMap | None = None,
) -> Path:
    """Copy exported files into a timestamped directory with manifest + HTML."""

    created = _dt.datetime.now(_dt.timezone.utc)
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = run_id[:8] or "run"
    run_root = base_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = _unique_run_dir(run_root / f"{created.strftime('%Y%m%d_%H%M%S_%f')}_{run_prefix}")
    run_dir.mkdir(parents=True, exist_ok=False)

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
    if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes, bytearray)):
        selected_list = list(selected)
    elif selected is None:
        selected_list = []
    else:
        selected_list = [selected]
    resolver = identity_map or (
        IdentityMap.from_config(config) if isinstance(config, Mapping) else IdentityMap()
    )
    selected_entities = _selected_entities(selected_list, resolver)

    manifest: dict[str, Any] = {
        "schema_version": "trend.run_artifacts/1",
        "run_id": run_id,
        "created": created.isoformat().replace("+00:00", "Z"),
        "config_path": config_path,
        "input_path": str(input_path),
        "config_sha256": sha256_config(config),
        "git_hash": _git_hash(),
        "data_window": _data_window(df),
        "data_reality": _data_reality(df),
        "metrics": _serialise_stats(details.get("out_ew_stats")) or _summarise_metrics(metrics_df),
        "metrics_overview": _summarise_metrics(metrics_df),
        "selected_funds": selected_list,
        "selected_entities": selected_entities,
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
    manifest["run_directory"] = str(run_dir.resolve())

    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(normalise_for_json(manifest), indent=2), encoding="utf-8")

    html_path = run_dir / "report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        _render_html(run_id=run_id, created=created, manifest=manifest, summary_text=summary_text),
        encoding="utf-8",
    )

    # Stable per-run-id index so a prior run for a given run_id is discoverable
    # regardless of its timestamped directory name. This is what powers the
    # CLI ``--skip-if-exists`` short-circuit.
    index_path = _run_index_path(base_dir, run_id)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_directory": str(run_dir.resolve()),
                "manifest": str(manifest_path.resolve()),
                "created": manifest["created"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return run_dir


def _run_index_path(base_dir: Path, run_id: str) -> Path:
    """Return the path of the stable per-run-id index pointer file."""

    return Path(base_dir) / "runs" / "index" / f"{run_id}.json"


def _unique_run_dir(base_path: Path) -> Path:
    """Return a non-existing run directory path based on *base_path*."""

    if not base_path.exists():
        return base_path
    suffix = 1
    while True:
        candidate = base_path.with_name(f"{base_path.name}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def find_existing_run(output_dir: Path | str, run_id: str) -> Path | None:
    """Return the manifest path for a previously completed run, if present.

    Looks up the stable per-run-id index written by :func:`write_run_artifacts`
    and returns the recorded manifest path when it still exists. Returns
    ``None`` when no prior run is recorded or its artifacts have been removed.
    """

    index_path = _run_index_path(Path(output_dir), run_id)
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    manifest = payload.get("manifest") if isinstance(payload, Mapping) else None
    if not manifest:
        return None
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path(output_dir) / manifest_path
    return manifest_path if manifest_path.exists() else None


__all__ = ["write_run_artifacts", "find_existing_run"]


def _selected_entities(
    selected_labels: Sequence[Any], identity_map: IdentityMap
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for raw_label in selected_labels:
        label = str(raw_label)
        entity = identity_map.resolve(label)
        item = by_id.get(entity.canonical_id)
        if item is None:
            by_id[entity.canonical_id] = entity.to_manifest(label=label, labels=[label])
            continue
        labels = item.setdefault("labels", [])
        if label not in labels:
            labels.append(label)
    return list(by_id.values())
