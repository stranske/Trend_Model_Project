#!/usr/bin/env python3
"""Diff two deterministic UI reproduction outputs.

This compares two folders created by ``scripts/reproduce_ui_run.py``.

It is intended for iterative debugging: after each change, re-run the
reproduction and diff against a known baseline to see what changed.

Usage:
  python scripts/diff_ui_runs.py \
    --a tmp/debug_runs/ui_run_2025-12-15_rebalance_weights_fix3 \
    --b tmp/debug_runs/ui_run_2025-12-15_after_underweight_ui \
    --out tmp/debug_runs/ui_run_2025-12-15_after_underweight_ui/diff

Outputs:
  - diff_summary.json (machine-readable)
  - diff_summary.md (human-readable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Directory A (baseline)")
    parser.add_argument("--b", required=True, help="Directory B (candidate)")
    parser.add_argument("--out", required=True, help="Output directory for diff")
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-10,
        help="Numeric tolerance for CSV comparison",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class JsonDiffItem:
    path: str
    a: Any
    b: Any


def _iter_json_diffs(a: Any, b: Any, *, prefix: str = "") -> Iterable[JsonDiffItem]:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for key in keys:
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a:
                yield JsonDiffItem(path=key_path, a=None, b=b[key])
                continue
            if key not in b:
                yield JsonDiffItem(path=key_path, a=a[key], b=None)
                continue
            yield from _iter_json_diffs(a[key], b[key], prefix=key_path)
        return

    if isinstance(a, list) and isinstance(b, list):
        # Keep list diffs coarse: if lists differ, report whole list.
        if a != b:
            yield JsonDiffItem(path=prefix or "<root>", a=a, b=b)
        return

    # Scalars / type mismatches
    if _is_number(a) and _is_number(b):
        if abs(float(a) - float(b)) > 0.0:
            yield JsonDiffItem(path=prefix or "<root>", a=a, b=b)
        return

    if a != b:
        yield JsonDiffItem(path=prefix or "<root>", a=a, b=b)


def _summarize_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
    }


def _diff_csv(a_path: Path, b_path: Path, *, tol: float) -> dict[str, Any]:
    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)

    summary: dict[str, Any] = {
        "a": {"rows": int(a_df.shape[0]), "cols": int(a_df.shape[1])},
        "b": {"rows": int(b_df.shape[0]), "cols": int(b_df.shape[1])},
        "same_shape": bool(a_df.shape == b_df.shape),
        "same_columns": bool(list(a_df.columns) == list(b_df.columns)),
        "column_diff": {
            "a_only": sorted(set(a_df.columns) - set(b_df.columns)),
            "b_only": sorted(set(b_df.columns) - set(a_df.columns)),
        },
        "numeric": {},
    }

    common_cols = [c for c in a_df.columns if c in b_df.columns]
    numeric_cols = [
        c
        for c in common_cols
        if pd.api.types.is_numeric_dtype(a_df[c]) and pd.api.types.is_numeric_dtype(b_df[c])
    ]

    # Compare numeric columns with tolerance (ignore NaNs)
    max_abs_by_col: dict[str, float] = {}
    changed_cells_by_col: dict[str, int] = {}
    for col in numeric_cols:
        a_s = pd.to_numeric(a_df[col], errors="coerce")
        b_s = pd.to_numeric(b_df[col], errors="coerce")
        diff = (a_s - b_s).abs()
        diff = diff[diff.notna()]
        if diff.empty:
            max_abs_by_col[col] = 0.0
            changed_cells_by_col[col] = 0
            continue
        max_abs = float(diff.max())
        changed = int((diff > tol).sum())
        max_abs_by_col[col] = max_abs
        changed_cells_by_col[col] = changed

    summary["numeric"] = {
        "tolerance": tol,
        "max_abs_diff_by_col": max_abs_by_col,
        "changed_cells_by_col": changed_cells_by_col,
    }
    return summary


def _write_markdown(out_path: Path, payload: dict[str, Any]) -> None:
    a_dir = payload["inputs"]["a"]
    b_dir = payload["inputs"]["b"]

    lines: list[str] = []
    lines.append("# UI Run Diff Summary")
    lines.append("")
    lines.append(f"- A: `{a_dir}`")
    lines.append(f"- B: `{b_dir}`")
    lines.append("")

    cfg = payload.get("config") or {}
    cfg_diffs = cfg.get("diffs") or []
    lines.append("## Config")
    if not cfg_diffs:
        lines.append("- No config differences detected.")
    else:
        lines.append(f"- Differences: {len(cfg_diffs)}")
        for item in cfg_diffs[:40]:
            path = item.get("path")
            lines.append(f"- `{path}`: A={item.get('a')!r} B={item.get('b')!r}")
        if len(cfg_diffs) > 40:
            lines.append(f"- (truncated; showing 40 of {len(cfg_diffs)})")
    lines.append("")

    lines.append("## Files")
    files = payload.get("files") or {}
    for name, info in files.items():
        if info.get("status") == "missing_in_b":
            lines.append(f"- `{name}`: missing in B")
        elif info.get("status") == "missing_in_a":
            lines.append(f"- `{name}`: missing in A")
        else:
            same = info.get("same")
            lines.append(f"- `{name}`: {'same' if same else 'different'}")
    lines.append("")

    lines.append("## CSV summaries")
    csvs = payload.get("csv") or {}
    if not csvs:
        lines.append("- No CSV comparisons available.")
    else:
        for name, info in csvs.items():
            lines.append(f"### {name}")
            lines.append(
                f"- same_shape={info.get('same_shape')} same_columns={info.get('same_columns')}"
            )
            numeric = info.get("numeric") or {}
            changed_cells_by_col = numeric.get("changed_cells_by_col") or {}
            total_changed = sum(int(v) for v in changed_cells_by_col.values())
            lines.append(f"- numeric_tol={numeric.get('tolerance')} changed_cells={total_changed}")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    a_dir = Path(args.a)
    b_dir = Path(args.b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        "config.json",
        "diagnostic.json",
        "metrics.csv",
        "period_results_summary.csv",
        "period_fund_weights.csv",
        "period_rebalance_weights.csv",
        "period_manager_changes.csv",
    ]

    files_payload: dict[str, Any] = {}
    for name in expected_files:
        a_path = a_dir / name
        b_path = b_dir / name
        if not a_path.exists() and not b_path.exists():
            continue
        if not a_path.exists():
            files_payload[name] = {"status": "missing_in_a"}
            continue
        if not b_path.exists():
            files_payload[name] = {"status": "missing_in_b"}
            continue
        a_hash = _sha256(a_path)
        b_hash = _sha256(b_path)
        files_payload[name] = {
            "status": "present",
            "same": bool(a_hash == b_hash),
            "a_sha256": a_hash,
            "b_sha256": b_hash,
        }

    config_payload: dict[str, Any] = {}
    a_cfg_path = a_dir / "config.json"
    b_cfg_path = b_dir / "config.json"
    if a_cfg_path.exists() and b_cfg_path.exists():
        a_cfg = _load_json(a_cfg_path)
        b_cfg = _load_json(b_cfg_path)
        diffs = list(_iter_json_diffs(a_cfg, b_cfg))
        config_payload = {
            "diff_count": len(diffs),
            "diffs": [
                {"path": d.path, "a": d.a, "b": d.b} for d in diffs if not d.path.startswith("repr")
            ],
        }

    csv_payload: dict[str, Any] = {}
    for name in [
        "metrics.csv",
        "period_results_summary.csv",
        "period_fund_weights.csv",
        "period_rebalance_weights.csv",
        "period_manager_changes.csv",
    ]:
        a_path = a_dir / name
        b_path = b_dir / name
        if not a_path.exists() or not b_path.exists():
            continue
        try:
            csv_payload[name] = _diff_csv(a_path, b_path, tol=float(args.tol))
        except Exception as exc:  # pragma: no cover
            csv_payload[name] = {"error": str(exc)}

    payload = {
        "inputs": {"a": str(a_dir), "b": str(b_dir)},
        "files": files_payload,
        "config": config_payload,
        "csv": csv_payload,
    }

    (out_dir / "diff_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown(out_dir / "diff_summary.md", payload)

    # Print a short console summary for quick iteration.
    cfg_count = int((payload.get("config") or {}).get("diff_count") or 0)
    print(f"Config diffs: {cfg_count}")
    different_files = [
        name
        for name, info in files_payload.items()
        if info.get("status") == "present" and not info.get("same")
    ]
    print(f"Different files: {len(different_files)}")
    for name in different_files:
        print(f"- {name}")
    print(f"Wrote: {out_dir / 'diff_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
