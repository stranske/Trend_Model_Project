#!/usr/bin/env python3
"""Reproduce a Streamlit UI run from a captured JSON payload.

This is a debugging harness: it runs the same config mapping as the Streamlit
app and writes a small set of baseline artifacts so future code changes can be
compared against a known reference.

Usage:
  source .venv/bin/activate
  python scripts/reproduce_ui_run.py \
    --params docs/debugging/ui_run_2025-12-15.json \
        --data "data/Trend Universe Data.csv" \
    --out tmp/debug_runs/ui_run_2025-12-15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure we can import the Streamlit app package when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.components.csv_validation import (  # noqa: E402
    CSVValidationError,
    DateCorrectionNeeded,
    validate_uploaded_csv,
)
from streamlit_app.components.date_correction import (  # noqa: E402
    apply_date_corrections,
)
from streamlit_app.components.data_cache import (  # noqa: E402
    load_dataset_from_bytes,
    load_dataset_from_path,
)
from streamlit_app.components.analysis_runner import (  # noqa: E402
    AnalysisPayload,
    _build_config,
    _prepare_returns,
)
from trend_analysis.api import run_simulation  # noqa: E402


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via error function (no scipy dependency)."""

    try:
        zf = float(z)
    except Exception:
        return float("nan")
    return 0.5 * (1.0 + math.erf(zf / math.sqrt(2.0)))


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_period_end(label: str) -> pd.Timestamp:
    """Period labels are slash-delimited; last component is the OOS end."""

    try:
        return pd.to_datetime(str(label).split("/")[-1], errors="coerce")
    except Exception:
        return pd.NaT


def _churn_from_rebalance_weights(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "period_rebalance_weights.csv"
    if not path.exists():
        return pd.DataFrame()

    reb = pd.read_csv(path)
    if "rebalance_date" not in reb.columns or "fund" not in reb.columns:
        return pd.DataFrame()

    reb["rebalance_date"] = pd.to_datetime(reb["rebalance_date"], errors="coerce")
    by_date = (
        reb.groupby("rebalance_date")["fund"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )

    ordered_dates = sorted(d for d in by_date.keys() if pd.notna(d))
    rows: list[dict[str, Any]] = []
    prev: set[str] | None = None
    for d in ordered_dates:
        cur = by_date[d]
        if prev is None:
            rows.append(
                {
                    "rebalance_date": pd.to_datetime(d).date().isoformat(),
                    "held_n": len(cur),
                    "adds_n": None,
                    "drops_n": None,
                    "total_changes_n": None,
                }
            )
            prev = cur
            continue

        adds = sorted(cur - prev)
        drops = sorted(prev - cur)
        rows.append(
            {
                "rebalance_date": pd.to_datetime(d).date().isoformat(),
                "year": int(pd.to_datetime(d).year),
                "held_n": len(cur),
                "adds_n": len(adds),
                "drops_n": len(drops),
                "total_changes_n": len(adds) + len(drops),
                "adds": "; ".join(adds),
                "drops": "; ".join(drops),
            }
        )
        prev = cur

    return pd.DataFrame(rows)


def _zscore_tail_sanity(
    selection_scores: pd.DataFrame, fund_weights: pd.DataFrame
) -> pd.DataFrame:
    """Compare observed z-score tail counts vs normal-tail expectations.

    This does NOT assert z-scores are normal; it is a heuristic to catch
    obvious breakages (e.g., z-scores not standardized, wrong frame used,
    or degenerate variance).
    """

    if selection_scores.empty or "period" not in selection_scores.columns:
        return pd.DataFrame()
    if "zscore" not in selection_scores.columns:
        return pd.DataFrame()

    held_by_period: dict[str, set[str]] = {}
    if not fund_weights.empty and {"period", "fund"}.issubset(fund_weights.columns):
        held_by_period = (
            fund_weights.groupby("period")["fund"]
            .apply(lambda s: set(s.dropna().astype(str)))
            .to_dict()
        )

    # Defaults align with the common UI settings.
    z_exit_soft = -0.5
    z_exit_hard = -1.5

    rows: list[dict[str, Any]] = []
    for period, dfp in selection_scores.groupby("period"):
        z = pd.to_numeric(dfp["zscore"], errors="coerce")
        universe_n = int(z.notna().sum())
        if universe_n <= 0:
            continue

        held = held_by_period.get(str(period), set())
        held_mask = (
            dfp["manager"].astype(str).isin(held)
            if held
            else pd.Series(False, index=dfp.index)
        )
        z_held = pd.to_numeric(dfp.loc[held_mask, "zscore"], errors="coerce")
        held_n = int(z_held.notna().sum())

        obs_soft = int((z < z_exit_soft).sum())
        obs_hard = int((z < z_exit_hard).sum())

        exp_soft = universe_n * _norm_cdf(z_exit_soft)
        exp_hard = universe_n * _norm_cdf(z_exit_hard)

        rows.append(
            {
                "period": period,
                "oos_end": _parse_period_end(str(period)).date().isoformat(),
                "year": int(_parse_period_end(str(period)).year),
                "universe_n": universe_n,
                "held_n": held_n,
                "z_exit_soft": z_exit_soft,
                "z_exit_hard": z_exit_hard,
                "obs_universe_z_lt_soft": obs_soft,
                "obs_universe_z_lt_hard": obs_hard,
                "exp_universe_z_lt_soft": exp_soft,
                "exp_universe_z_lt_hard": exp_hard,
                "obs_over_exp_soft": (
                    (obs_soft / exp_soft) if exp_soft > 0 else float("nan")
                ),
                "obs_over_exp_hard": (
                    (obs_hard / exp_hard) if exp_hard > 0 else float("nan")
                ),
                "held_min_z": _safe_float(z_held.min()) if held_n else float("nan"),
                "held_p10_z": (
                    _safe_float(z_held.quantile(0.10)) if held_n else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="Path to captured UI JSON")
    parser.add_argument("--data", required=True, help="Path to Trend Universe CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--threshold-metric",
        default=None,
        help="Override threshold-hold selection metric (e.g. Sharpe).",
    )
    return parser.parse_args()


def _load_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "model_state" not in payload:
        raise ValueError("params missing model_state")
    return payload


def _load_returns(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()

    # Reuse the Streamlit upload validator / date-correction pipeline so the
    # reproduction matches the app behaviour.
    try:
        validate_uploaded_csv(raw, required_columns=("Date",), max_rows=0)
    except CSVValidationError as exc:
        correction: DateCorrectionNeeded | None = getattr(exc, "date_correction", None)
        if correction is None:
            raise
        if correction.unfixable:
            example = correction.unfixable[0]
            raise ValueError(
                "Dataset contains unfixable dates; cannot auto-correct in reproduction harness. "
                f"Example: row {example[0] + 1} value={example[1]!r}"
            )

        df_raw = pd.read_csv(pd.io.common.BytesIO(correction.raw_data))
        df_fixed = apply_date_corrections(
            df_raw,
            correction.date_column,
            correction.corrections,
            drop_rows=correction.all_rows_to_drop,
        )

        corrected_bytes = df_fixed.to_csv(index=False).encode("utf-8")

        # Re-validate with corrected data (mirrors the Streamlit flow).
        validate_uploaded_csv(corrected_bytes, required_columns=("Date",), max_rows=0)

        validated_df, _meta = load_dataset_from_bytes(
            corrected_bytes, correction.original_name
        )
        return validated_df

    validated_df, _meta = load_dataset_from_path(str(path))
    return validated_df


def _filter_columns(
    df: pd.DataFrame,
    *,
    fund_columns: list[str],
    benchmark: str | None,
    risk_free: str | None,
) -> pd.DataFrame:
    keep = []
    if risk_free:
        keep.append(risk_free)
    if benchmark:
        keep.append(benchmark)
    keep.extend(fund_columns)

    existing = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in dataset: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    return df[existing]


def main() -> int:
    args = _parse_args()
    params_path = Path(args.params)
    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = _load_params(params_path)
    fund_columns = list(params.get("fund_columns") or [])
    benchmark = params.get("selected_benchmark")
    risk_free = params.get("selected_risk_free")

    model_state = dict(params["model_state"])
    # Streamlit config mapping looks for `risk_free_column` inside model_state.
    if risk_free and not model_state.get("risk_free_column"):
        model_state["risk_free_column"] = risk_free

    returns = _load_returns(data_path)
    returns = _filter_columns(
        returns, fund_columns=fund_columns, benchmark=benchmark, risk_free=risk_free
    )

    payload = AnalysisPayload(
        returns=returns, model_state=model_state, benchmark=benchmark
    )
    cfg = _build_config(payload)

    # Optional: override the threshold-hold selection metric for repro runs.
    # This is useful for auditing the non-blended baseline (e.g., Sharpe-only).
    if args.threshold_metric:
        try:
            metric = str(args.threshold_metric).strip()
        except Exception:
            metric = ""
        if metric:
            port = getattr(cfg, "portfolio", {}) or {}
            th = port.get("threshold_hold") if isinstance(port, dict) else None
            if isinstance(th, dict):
                th = dict(th)
                th["metric"] = metric
                # If we are no longer running blended, remove blended weights to
                # avoid implying they affect the run.
                if metric.lower() != "blended":
                    th.pop("blended_weights", None)

                # Keep selector seeding consistent with the overridden metric.
                selector = port.get("selector") if isinstance(port, dict) else None
                if isinstance(selector, dict):
                    selector = dict(selector)
                    params = selector.get("params")
                    if isinstance(params, dict):
                        params = dict(params)
                    else:
                        params = {}
                    params["rank_column"] = metric
                    selector["params"] = params
                else:
                    selector = {"name": "rank", "params": {"rank_column": metric}}

                port = dict(port)
                port["threshold_hold"] = th
                port["selector"] = selector
                # Prefer pydantic's model_copy/update path when available.
                model_copy = getattr(cfg, "model_copy", None)
                if callable(model_copy):
                    cfg = model_copy(update={"portfolio": port})
                else:
                    setattr(cfg, "portfolio", port)

    result = run_simulation(cfg, _prepare_returns(returns))

    # Persist a minimal baseline bundle (avoid pickling large/un-stable objects).
    def _dump_cfg(obj: Any) -> str:
        if hasattr(obj, "model_dump"):
            try:
                return json.dumps(obj.model_dump(), indent=2, default=str)
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return json.dumps(obj.dict(), indent=2, default=str)
            except Exception:
                pass
        if hasattr(obj, "json"):
            try:
                return obj.json(indent=2)
            except TypeError:
                return obj.json()
            except Exception:
                pass
        return json.dumps({"repr": repr(obj)}, indent=2, default=str)

    (out_dir / "config.json").write_text(_dump_cfg(cfg), encoding="utf-8")

    # Always persist something diagnostic-like, even when multi-period
    # exceptions were swallowed inside the API.
    diagnostic = getattr(result, "diagnostic", None)
    diag_payload: dict[str, Any] | None = None
    if diagnostic is not None:
        diag_payload = {
            "reason_code": getattr(diagnostic, "reason_code", None),
            "message": getattr(diagnostic, "message", None),
            "context": getattr(diagnostic, "context", None),
        }
    else:
        details = getattr(result, "details", None) or {}
        error = details.get("error") if isinstance(details, dict) else None
        if error:
            diag_payload = {"error": error}

    (out_dir / "diagnostic.json").write_text(
        json.dumps(diag_payload, indent=2, default=str), encoding="utf-8"
    )

    # Metrics (top-level API output)
    if isinstance(result.metrics, pd.DataFrame):
        result.metrics.to_csv(out_dir / "metrics.csv")

    # Details are heterogeneous; save only stable slices if present.
    details = result.details or {}

    # 1) Manager changes / churn
    mgr_changes = details.get("manager_changes")
    if isinstance(mgr_changes, list):
        (out_dir / "manager_changes.json").write_text(
            json.dumps(mgr_changes, indent=2, default=str), encoding="utf-8"
        )

    # 2) Period-level holdings + weights
    periods = details.get("period_results")
    if isinstance(periods, list):
        summary_rows: list[dict[str, Any]] = []
        fund_weight_rows: list[dict[str, Any]] = []
        ew_weight_rows: list[dict[str, Any]] = []
        rebalance_weight_rows: list[dict[str, Any]] = []
        manager_change_rows: list[dict[str, Any]] = []
        selection_score_rows: list[dict[str, Any]] = []
        for item in periods:
            if not isinstance(item, dict):
                continue

            period = item.get("period")
            period_label = item.get("label")
            if not period_label and isinstance(period, (list, tuple)):
                period_label = "/".join(str(x) for x in period)

            fund_weights = item.get("fund_weights")
            if isinstance(fund_weights, dict):
                for fund, weight in fund_weights.items():
                    try:
                        w = float(weight)
                    except Exception:
                        continue
                    if abs(w) <= 1e-12:
                        continue
                    fund_weight_rows.append(
                        {
                            "period": period_label,
                            "fund": fund,
                            "weight": w,
                        }
                    )
            ew_weights = item.get("ew_weights")
            if isinstance(ew_weights, dict):
                for fund, weight in ew_weights.items():
                    try:
                        w = float(weight)
                    except Exception:
                        continue
                    if abs(w) <= 1e-12:
                        continue
                    ew_weight_rows.append(
                        {
                            "period": period_label,
                            "fund": fund,
                            "weight": w,
                        }
                    )

            rebalance_weights = item.get("rebalance_weights")
            if (
                isinstance(rebalance_weights, pd.DataFrame)
                and not rebalance_weights.empty
            ):
                for reb_date, row in rebalance_weights.iterrows():
                    stamp = pd.to_datetime(reb_date).date().isoformat()
                    for fund, weight in row.items():
                        try:
                            w = float(weight)
                        except Exception:
                            continue
                        if abs(w) <= 1e-12:
                            continue
                        rebalance_weight_rows.append(
                            {
                                "period": period_label,
                                "rebalance_date": stamp,
                                "fund": str(fund),
                                "weight": w,
                            }
                        )

            mgr_changes_period = item.get("manager_changes")
            if isinstance(mgr_changes_period, list):
                for change in mgr_changes_period:
                    if isinstance(change, dict):
                        manager_change_rows.append({"period": period_label, **change})

            selection_scores = item.get("selection_score_frame")
            if (
                isinstance(selection_scores, pd.DataFrame)
                and not selection_scores.empty
            ):
                metric = item.get("selection_metric")
                for manager, row in selection_scores.iterrows():
                    payload: dict[str, Any] = {
                        "period": period_label,
                        "manager": str(manager),
                        "selection_metric": metric,
                    }
                    # Export full selection-score frame columns so downstream
                    # audits can reproduce z-score calculations (metric value +
                    # universe mean/std) without re-running the simulation.
                    for col, val in row.to_dict().items():
                        # Preserve column names as-is.
                        payload[str(col)] = val
                    selection_score_rows.append(payload)

            summary_rows.append(
                {
                    "label": period_label,
                    "selected_funds_n": len(item.get("selected_funds") or []),
                    "fund_weights_n": len(item.get("fund_weights") or {}),
                    "ew_weights_n": len(item.get("ew_weights") or {}),
                    "turnover": item.get("turnover"),
                    "transaction_cost": item.get("transaction_cost"),
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            out_dir / "period_results_summary.csv", index=False
        )
        if fund_weight_rows:
            pd.DataFrame(fund_weight_rows).to_csv(
                out_dir / "period_fund_weights.csv", index=False
            )
        if ew_weight_rows:
            pd.DataFrame(ew_weight_rows).to_csv(
                out_dir / "period_ew_weights.csv", index=False
            )
        if rebalance_weight_rows:
            pd.DataFrame(rebalance_weight_rows).to_csv(
                out_dir / "period_rebalance_weights.csv", index=False
            )
        if manager_change_rows:
            pd.DataFrame(manager_change_rows).to_csv(
                out_dir / "period_manager_changes.csv", index=False
            )
        if selection_score_rows:
            pd.DataFrame(selection_score_rows).to_csv(
                out_dir / "period_selection_scores.csv", index=False
            )

    # --- Sanity heuristics -------------------------------------------------
    # These are fast checks intended to catch obviously suspicious outputs
    # early (e.g., broken z-score standardisation or implausible churn).
    sanity: dict[str, Any] = {}

    try:
        max_changes = int(cfg.portfolio.get("turnover_budget_max_changes") or 0)
    except Exception:
        max_changes = 0
    sanity["max_changes_per_rebalance"] = max_changes

    churn = _churn_from_rebalance_weights(out_dir)
    if not churn.empty:
        churn.to_csv(out_dir / "sanity_rebalance_churn.csv", index=False)

        steps = int(churn["total_changes_n"].dropna().shape[0])
        adds_total = int(
            pd.to_numeric(churn["adds_n"], errors="coerce").fillna(0).sum()
        )
        drops_total = int(
            pd.to_numeric(churn["drops_n"], errors="coerce").fillna(0).sum()
        )
        sanity.update(
            {
                "rebalance_steps": steps,
                "adds_total": adds_total,
                "drops_total": drops_total,
                "total_changes": adds_total + drops_total,
                "avg_total_changes_per_rebalance": (
                    (adds_total + drops_total) / steps if steps else None
                ),
            }
        )

        if max_changes > 0:
            viol = churn[
                pd.to_numeric(churn["total_changes_n"], errors="coerce").fillna(0)
                > max_changes
            ]
            sanity["rebalances_total_changes_over_max"] = int(viol.shape[0])
            if not viol.empty:
                viol.to_csv(
                    out_dir
                    / f"sanity_rebalance_churn_violations_over_{max_changes}.csv",
                    index=False,
                )

    # Z-score tail sanity (universe-level)
    try:
        sel_path = out_dir / "period_selection_scores.csv"
        w_path = out_dir / "period_fund_weights.csv"
        if sel_path.exists():
            sel = pd.read_csv(sel_path)
            fw = pd.read_csv(w_path) if w_path.exists() else pd.DataFrame()
            tail = _zscore_tail_sanity(sel, fw)
            if not tail.empty:
                tail = tail.sort_values("oos_end", kind="stable")
                tail.to_csv(out_dir / "sanity_zscore_tail_by_period.csv", index=False)

                # Flag obvious anomalies: tail frequency wildly off.
                # We use very wide bands to avoid false positives.
                soft_ratios = pd.to_numeric(tail["obs_over_exp_soft"], errors="coerce")
                hard_ratios = pd.to_numeric(tail["obs_over_exp_hard"], errors="coerce")
                sanity["tail_soft_ratio_median"] = _safe_float(soft_ratios.median())
                sanity["tail_hard_ratio_median"] = _safe_float(hard_ratios.median())
                sanity["tail_soft_ratio_p10"] = _safe_float(soft_ratios.quantile(0.10))
                sanity["tail_soft_ratio_p90"] = _safe_float(soft_ratios.quantile(0.90))
                sanity["tail_hard_ratio_p10"] = _safe_float(hard_ratios.quantile(0.10))
                sanity["tail_hard_ratio_p90"] = _safe_float(hard_ratios.quantile(0.90))

                # Wide, heuristic guardrails (intentionally permissive).
                sanity_warnings: list[str] = []
                if pd.notna(sanity["tail_soft_ratio_median"]) and (
                    sanity["tail_soft_ratio_median"] < 0.1
                    or sanity["tail_soft_ratio_median"] > 10.0
                ):
                    sanity_warnings.append(
                        "Universe z-score soft-tail frequency looks suspicious (median obs/exp outside [0.1, 10])."
                    )
                if pd.notna(sanity["tail_hard_ratio_median"]) and (
                    sanity["tail_hard_ratio_median"] < 0.1
                    or sanity["tail_hard_ratio_median"] > 10.0
                ):
                    sanity_warnings.append(
                        "Universe z-score hard-tail frequency looks suspicious (median obs/exp outside [0.1, 10])."
                    )

                if sanity_warnings:
                    sanity["warnings"] = sanity_warnings
    except Exception as exc:
        sanity["sanity_error"] = f"{type(exc).__name__}: {exc}"

    (out_dir / "sanity_summary.json").write_text(
        json.dumps(sanity, indent=2, default=str), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
