#!/usr/bin/env python3
"""Streamlit parameter sweep with LLM comparisons.

Runs baseline + variant simulations for selected parameters, writes JSON configs
and outputs per test, and emits LLM-based comparison notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app import state as app_state  # noqa: E402
from streamlit_app.components import analysis_runner, comparison_llm  # noqa: E402
from streamlit_app.components.data_schema import load_and_validate_file  # noqa: E402
from streamlit_app.components.llm_settings import (  # noqa: E402
    default_api_key,
    resolve_llm_provider_config,
)
from trend_analysis.export import (  # noqa: E402
    metrics_from_result,
    summary_frame_from_result,
)
from trend_analysis.llm import create_llm  # noqa: E402

DEFAULT_LLM_QUESTIONS = """Evaluate whether the simulation outcome differences match intuition.
Focus on whether the parameter change should increase/decrease risk, turnover, and performance.
Call out any results that look counterintuitive or suggest a wiring bug.
"""

ADVISOR_SYSTEM_PROMPT = """You are an expert quantitative analyst reviewing parameter sensitivity tests.
Return a JSON object with:
- adequate: true/false
- rationale: short string
- suggested_values: list of 2 numeric values (if not adequate)
Only return JSON. No extra text.
"""


@dataclass(frozen=True)
class NumericSpec:
    delta_pct: float
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class TestCase:
    name: str
    baseline_value: Any
    test_value: Any
    label: str
    category: str


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    return json.loads(text)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _hash_df(df: pd.DataFrame) -> str:
    payload = df.to_json(date_format="iso", date_unit="ns", orient="split")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _load_dataset(path: Path) -> pd.DataFrame:
    df, _meta = load_and_validate_file(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(axis=0, how="any")
    return df


def _filter_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in df.columns]
    return df[cols].copy()


def _serialise_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "columns": [str(c) for c in value.columns],
            "rows": value.to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        return {
            "type": "series",
            "index": [str(i) for i in value.index],
            "values": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(k): _serialise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise_value(v) for v in value]
    return value


def _serialise_result(result: Any) -> dict[str, Any]:
    details = getattr(result, "details", None)
    metrics = getattr(result, "metrics", None)
    payload: dict[str, Any] = {
        "details": (
            _serialise_value(details) if isinstance(details, Mapping) else details
        ),
        "metrics": (
            _serialise_value(metrics) if isinstance(metrics, pd.DataFrame) else metrics
        ),
    }
    if isinstance(details, Mapping):
        payload["summary"] = _serialise_value(summary_frame_from_result(details))
        payload["metrics_frame"] = _serialise_value(metrics_from_result(details))
    return payload


def _ensure_numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _clamp_numeric(value: float, spec: NumericSpec) -> float:
    if spec.min_value is not None:
        value = max(value, spec.min_value)
    if spec.max_value is not None:
        value = min(value, spec.max_value)
    return value


def _build_numeric_values(base_value: Any, spec: NumericSpec) -> list[float]:
    base = _ensure_numeric(base_value)
    if math.isnan(base):
        return []
    delta = base * spec.delta_pct
    low = _clamp_numeric(base - delta, spec)
    high = _clamp_numeric(base + delta, spec)
    values = [low, high]
    return [v for v in values if not math.isnan(v)]


def _run_llm_advisor(
    *,
    parameter: str,
    baseline: float,
    values: list[float],
    provider: str,
    api_key: str,
) -> dict[str, Any]:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = (
        f"Parameter: {parameter}\n"
        f"Baseline value: {baseline}\n"
        f"Test values: {values}\n\n"
        "Are these values adequate to evaluate sensitivity?"
    )
    config = resolve_llm_provider_config(provider, api_key=api_key)
    llm = create_llm(config)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", ADVISOR_SYSTEM_PROMPT),
            ("user", "{prompt}"),
        ]
    )
    chain = template | llm
    response = chain.invoke({"prompt": prompt})
    text = getattr(response, "content", None) or str(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"adequate": False, "rationale": "Invalid JSON response", "raw": text}


def _compare_with_llm(
    *,
    baseline: Any,
    variant: Any,
    diff_text: str,
    label: str,
    provider: str,
    api_key: str,
) -> dict[str, Any]:
    result = comparison_llm.generate_comparison_explanation(
        baseline.details,
        variant.details,
        config_diff=diff_text,
        questions=DEFAULT_LLM_QUESTIONS,
        label_a="Baseline",
        label_b=label,
        provider=provider,
        api_key=api_key,
    )
    return {
        "text": result.text,
        "trace_url": result.trace_url,
        "metric_count": result.metric_count,
        "created_at": result.created_at,
        "claim_issues": [
            {"kind": issue.kind, "message": issue.message, "detail": issue.detail}
            for issue in result.claim_issues
        ],
        "questions": DEFAULT_LLM_QUESTIONS,
    }


def _zip_folder(source: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source))


def _build_test_cases(
    base_state: Mapping[str, Any], spec: dict[str, Any]
) -> list[TestCase]:
    cases: list[TestCase] = []
    from scripts.test_settings_wiring import SETTINGS_TO_TEST

    for setting in SETTINGS_TO_TEST:
        cases.append(
            TestCase(
                name=setting.name,
                baseline_value=setting.baseline_value,
                test_value=setting.test_value,
                label=f"{setting.name}={setting.test_value}",
                category=setting.category,
            )
        )

    categorical = spec.get("categorical", {}) if isinstance(spec, Mapping) else {}
    for name, options in categorical.items():
        if not isinstance(options, list):
            continue
        baseline = base_state.get(name)
        for option in options:
            if option == baseline:
                continue
            cases.append(
                TestCase(
                    name=name,
                    baseline_value=baseline,
                    test_value=option,
                    label=f"{name}={option}",
                    category="categorical",
                )
            )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--enable-advisor", action="store_true")
    parser.add_argument("--advisor-max-iterations", type=int, default=1)
    args = parser.parse_args()

    base_config = _load_config(Path(args.base_config))
    spec = _load_config(Path(args.spec))

    df = _load_dataset(Path(args.dataset))
    df = _filter_columns(df, base_config.get("analysis_fund_columns", []))
    if df.empty:
        raise SystemExit("Filtered dataset is empty.")
    data_hash = _hash_df(df)

    model_state = base_config.get("model_state", {})
    benchmark = base_config.get("selected_benchmark")

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = args.llm_provider
    api_key = default_api_key(provider) if args.enable_llm else None
    if args.enable_llm and not api_key:
        raise SystemExit("LLM enabled but no API key was resolved.")

    cache: dict[str, Any] = {}

    def run_cached(state: Mapping[str, Any]) -> Any:
        key = json.dumps(state, sort_keys=True, default=str)
        if key not in cache:
            cache[key] = analysis_runner.run_analysis(
                df, state, benchmark, data_hash=data_hash
            )
        return cache[key]

    numeric_spec = spec.get("numeric", {}) if isinstance(spec, Mapping) else {}

    for test in _build_test_cases(model_state, spec):
        baseline_state = dict(model_state)
        if test.baseline_value is not None:
            baseline_state[test.name] = test.baseline_value
        variant_state = dict(baseline_state)
        variant_state[test.name] = test.test_value

        base_result = run_cached(baseline_state)
        variant_result = run_cached(variant_state)

        diff_text = app_state.format_model_state_diff(
            app_state.diff_model_states(baseline_state, variant_state),
            label_a="Baseline",
            label_b=test.label,
        )

        test_dir = output_dir / test.name / test.label
        _write_json(test_dir / "config_baseline.json", baseline_state)
        _write_json(test_dir / "config_variant.json", variant_state)
        _write_json(test_dir / "result_baseline.json", _serialise_result(base_result))
        _write_json(test_dir / "result_variant.json", _serialise_result(variant_result))
        (test_dir / "config_diff.txt").write_text(diff_text)

        if args.enable_llm and api_key:
            llm_payload = _compare_with_llm(
                baseline=base_result,
                variant=variant_result,
                diff_text=diff_text,
                label=test.label,
                provider=provider,
                api_key=api_key,
            )
            _write_json(test_dir / "comparison_llm.json", llm_payload)
            (test_dir / "comparison_llm.txt").write_text(llm_payload["text"])

        spec_entry = numeric_spec.get(test.name)
        if args.enable_advisor and isinstance(spec_entry, Mapping):
            base_value = _ensure_numeric(baseline_state.get(test.name))
            spec_obj = NumericSpec(
                delta_pct=float(spec_entry.get("delta_pct", 0.2)),
                min_value=spec_entry.get("min"),
                max_value=spec_entry.get("max"),
            )
            values = _build_numeric_values(base_value, spec_obj)
            if values and args.enable_llm and api_key:
                advisor = _run_llm_advisor(
                    parameter=test.name,
                    baseline=base_value,
                    values=values,
                    provider=provider,
                    api_key=api_key,
                )
                _write_json(test_dir / "advisor.json", advisor)

    _zip_folder(output_dir, Path(args.zip_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
