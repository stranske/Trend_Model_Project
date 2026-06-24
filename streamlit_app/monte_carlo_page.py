"""Monte Carlo simulation page for the Streamlit application."""

from __future__ import annotations

import copy
import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_app.components.data_cache import cache_key_for_frame
from streamlit_app.components.guardrails import infer_frequency
from streamlit_app.components.progress_eta import progress_ratio_and_remaining
from trend_analysis.config.ui_mapping import build_config_from_ui_state
from trend_analysis.monte_carlo.aggregator import aggregate_monte_carlo_results
from trend_analysis.monte_carlo.export_bundle import save as save_chart_bundle
from trend_analysis.monte_carlo.registry import (
    ScenarioRegistryEntry,
    list_scenarios,
    load_scenario,
)
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings
from trend_analysis.viz.adapters import (
    make_paths,
    make_summary,
)


def _should_auto_render() -> bool:
    """Return True when running inside an active Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


MC_RESULTS_KEY = "mc_results"
MC_RUNNING_KEY = "mc_running"
MC_CANCEL_KEY = "mc_cancel_requested"
MC_LAST_ERROR_KEY = "mc_last_error"
MC_LAST_VALIDATION_KEY = "mc_last_validation"


def _session_frequency(returns: pd.DataFrame) -> str:
    meta = st.session_state.get("schema_meta")
    if isinstance(meta, Mapping):
        freq = meta.get("frequency_code") or meta.get("frequency")
        if isinstance(freq, str) and freq.strip():
            return freq.strip().upper()
    return infer_frequency(returns.index)


def _session_csv_path() -> str | None:
    for key in ("data_saved_path", "uploaded_file_path"):
        candidate = st.session_state.get(key)
        if isinstance(candidate, str) and candidate:
            path = Path(candidate)
            if path.exists() and path.suffix.lower() == ".csv":
                return str(path)
    return None


def _analysis_frame_from_session(
    returns: pd.DataFrame,
    model_state: Mapping[str, Any],
    benchmark: str | None,
) -> pd.DataFrame:
    applied_funds = st.session_state.get("analysis_fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = st.session_state.get("fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = []

    selected_rf = st.session_state.get("selected_risk_free")
    info_ratio_benchmark = model_state.get("info_ratio_benchmark")
    regime_proxy = None
    if bool(model_state.get("regime_enabled", False)):
        regime_proxy = model_state.get("regime_proxy")
    prohibited = {selected_rf, benchmark, info_ratio_benchmark, regime_proxy} - {None}

    sanitized_funds = [
        c for c in applied_funds if c in returns.columns and c not in prohibited
    ]
    keep_cols = list(sanitized_funds)
    for extra in (selected_rf, benchmark, regime_proxy):
        if extra and extra in returns.columns and extra not in keep_cols:
            keep_cols.append(extra)
    return returns[keep_cols].copy() if keep_cols else returns.copy()


def _returns_to_price_history(returns: pd.DataFrame) -> pd.DataFrame:
    numeric = returns.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError("session returns contain no numeric columns")
    if (numeric <= -1.0).any().any():
        raise ValueError("session returns contain values <= -1")
    return (1.0 + numeric).cumprod() * 100.0


def _session_runner_kwargs() -> tuple[dict[str, Any], str | None]:
    returns = st.session_state.get("returns_df")
    model_state = st.session_state.get("model_state")
    if not isinstance(returns, pd.DataFrame) or not isinstance(model_state, Mapping):
        return {}, None

    benchmark = st.session_state.get("selected_benchmark")
    selected_rf = st.session_state.get("selected_risk_free")
    effective_model_state = dict(model_state)
    if selected_rf:
        effective_model_state["risk_free_column"] = selected_rf

    try:
        analysis_frame = _analysis_frame_from_session(
            returns,
            effective_model_state,
            benchmark if isinstance(benchmark, str) else None,
        )
        base_config = build_config_from_ui_state(
            returns=analysis_frame,
            model_state=effective_model_state,
            benchmark=benchmark if isinstance(benchmark, str) else None,
            frequency=_session_frequency(analysis_frame),
            csv_path=_session_csv_path(),
        )
        if isinstance(benchmark, str) and benchmark in analysis_frame.columns:
            indices = list(base_config.portfolio.get("indices_list") or [])
            if benchmark not in indices:
                base_config.portfolio["indices_list"] = [*indices, benchmark]
        price_history = _returns_to_price_history(analysis_frame)
    except Exception as exc:
        return {}, str(exc)
    return {"base_config": base_config, "price_history": price_history}, None


class _RunCancelled(RuntimeError):
    """Raised when the user cancels a Monte Carlo run."""


def _cache_data(*args: object, **kwargs: object):
    cache_data = getattr(st, "cache_data", None)
    if callable(cache_data):
        return cache_data(*args, **kwargs)

    def _identity(func):
        return func

    return _identity


@_cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache_key_for_frame})
def _cached_make_summary(
    results_frame: pd.DataFrame, fold_selection: int | str | None
) -> pd.DataFrame:
    return make_summary(results_frame, fold_selection=fold_selection)


@_cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache_key_for_frame})
def _cached_make_paths(nav_paths: pd.DataFrame) -> pd.DataFrame:
    return make_paths(nav_paths)


def _collect_tags(entries: Iterable[ScenarioRegistryEntry]) -> list[str]:
    tags: set[str] = set()
    for entry in entries:
        tags.update(entry.tags)
    return sorted(tags)


def _scenario_lookup(
    entries: Iterable[ScenarioRegistryEntry],
) -> dict[str, ScenarioRegistryEntry]:
    return {entry.name: entry for entry in entries}


def _clamp_int(value: int | None, minimum: int, maximum: int) -> int:
    if value is None:
        return minimum
    return max(min(int(value), maximum), minimum)


def _coerce_seed(text: str) -> tuple[int | None, str | None]:
    cleaned = text.strip()
    if not cleaned:
        return None, None
    try:
        seed = int(cleaned)
    except ValueError:
        return None, "Random seed must be an integer."
    if seed < 0:
        return None, "Random seed must be a non-negative integer."
    return seed, None


def _build_override_scenario(
    scenario: MonteCarloScenario,
    *,
    n_paths: int,
    horizon_years: int,
    seed: int | None,
    jobs: int,
) -> MonteCarloScenario:
    settings = scenario.monte_carlo
    if not isinstance(settings, MonteCarloSettings):
        raise TypeError("monte_carlo settings are not resolved")
    updated_settings = MonteCarloSettings(
        mode=settings.mode,
        n_paths=n_paths,
        horizon_years=float(horizon_years),
        frequency=settings.frequency,
        seed=seed,
        jobs=jobs,
    )
    updated_scenario = copy.copy(scenario)
    updated_scenario.monte_carlo = updated_settings
    return updated_scenario


def _fold_options(scenario: MonteCarloScenario) -> list[str]:
    if not scenario.enable_fold_runs:
        return []
    folds = scenario.folds
    if not isinstance(folds, Mapping):
        return []
    if folds.get("enabled", True) is False:
        return []
    count: int | None = None
    if "n_folds" in folds:
        try:
            count = int(folds["n_folds"])
        except (TypeError, ValueError):
            count = None
    elif "fold_starts" in folds:
        starts = folds.get("fold_starts")
        if isinstance(starts, (list, tuple)):
            count = len(starts)
    options = ["All folds"]
    if count:
        options.extend([f"Fold {idx}" for idx in range(1, count + 1)])
    return options


def _filter_results_by_fold(
    results: pd.DataFrame, selection: str | None
) -> pd.DataFrame:
    if not selection or selection == "All folds":
        return results
    if results.empty:
        return results
    if "fold_id" in results.columns:
        tokens = selection.split()
        if tokens and tokens[-1].isdigit():
            fold_id = int(tokens[-1])
            return results[results["fold_id"] == fold_id]
    if "fold_label" in results.columns:
        return results[results["fold_label"] == selection]
    return results


def _extract_nav_paths(results: object, *, fold_id: int | None = None) -> pd.DataFrame:
    if hasattr(results, "nav_paths"):
        nav_paths = getattr(results, "nav_paths")
        if isinstance(nav_paths, pd.DataFrame):
            return nav_paths
    if hasattr(results, "metadata"):
        metadata = getattr(results, "metadata")
        if isinstance(metadata, Mapping):
            if fold_id is not None:
                nav_paths_by_fold = metadata.get("nav_paths_by_fold")
                if isinstance(nav_paths_by_fold, Mapping):
                    nav_paths = nav_paths_by_fold.get(fold_id)
                    if isinstance(nav_paths, pd.DataFrame):
                        return nav_paths
            nav_paths = metadata.get("nav_paths")
            if isinstance(nav_paths, pd.DataFrame):
                return nav_paths
    if isinstance(results, Mapping):
        if fold_id is not None:
            nav_paths_by_fold = results.get("nav_paths_by_fold")
            if isinstance(nav_paths_by_fold, Mapping):
                nav_paths = nav_paths_by_fold.get(fold_id)
                if isinstance(nav_paths, pd.DataFrame):
                    return nav_paths
        nav_paths = results.get("nav_paths")
        if isinstance(nav_paths, pd.DataFrame):
            return nav_paths
    return pd.DataFrame()


def _fold_selection_for_adapters(selection: str | None) -> int | str | None:
    if not selection or selection == "All folds":
        return None
    tokens = selection.split()
    if tokens and tokens[-1].isdigit():
        return int(tokens[-1])
    return selection


def _render_diagnostic_charts(
    summary: pd.DataFrame, paths: pd.DataFrame
) -> dict[str, go.Figure]:
    charts: dict[str, go.Figure] = {}
    if summary.empty:
        st.warning("Diagnostics unavailable: summary frame is empty.")
        return charts
    if paths.empty:
        st.warning("Diagnostics unavailable: canonical paths are empty.")
        return charts

    from trend_analysis.viz import sharpe_ladder as sharpe_ladder_chart
    from trend_analysis.viz.charts import corr_heatmap as corr_heatmap_chart
    from trend_analysis.viz.charts import rolling_panel as rolling_panel_chart
    from trend_analysis.viz.charts import (
        seasonality_heatmap as seasonality_heatmap_chart,
    )

    try:
        sharpe_fig = sharpe_ladder_chart.make(summary, metric="sharpe")
    except Exception:
        st.warning(
            "Sharpe ladder unavailable: summary does not include a usable 'sharpe' metric."
        )
        sharpe_fig = go.Figure()
    corr_fig = corr_heatmap_chart.build_figure(paths, window=60)
    rolling_fig = rolling_panel_chart.build_figure(
        paths, window=12, periods_per_year=12, max_paths=6
    )
    seasonality_fig = seasonality_heatmap_chart.build_figure(paths)
    charts = {
        "Sharpe Ladder": sharpe_fig,
        "Correlation Heatmap": corr_fig,
        "Rolling Diagnostics": rolling_fig,
        "Seasonality Heatmap": seasonality_fig,
    }

    st.plotly_chart(sharpe_fig, use_container_width=True)
    st.plotly_chart(corr_fig, use_container_width=True)
    st.plotly_chart(rolling_fig, use_container_width=True)
    st.plotly_chart(seasonality_fig, use_container_width=True)
    return charts


def _progress_callback_factory(
    *,
    progress_bar: object,
    elapsed_slot: object,
    eta_slot: object,
    start_time: float,
) -> callable:
    last_update = 0.0

    def _update(payload: Mapping[str, object]) -> None:
        nonlocal last_update
        if st.session_state.get(MC_CANCEL_KEY):
            raise _RunCancelled("Monte Carlo run cancelled")
        completed = int(payload.get("completed", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        now = monotonic()
        if completed < total and (now - last_update) < 1.0:
            return
        last_update = now
        elapsed = max(now - start_time, 0.0)
        ratio = (completed / total) if total else 0.0
        remaining = 0.0
        if ratio > 0:
            ratio, remaining = progress_ratio_and_remaining(elapsed, elapsed / ratio)
        progress_text = f"Running paths ({completed}/{total})"
        if total:
            progress_text = f"Running paths ({completed}/{total})"
        try:
            progress_bar.progress(min(ratio, 1.0), text=progress_text)
        except Exception:
            progress_bar.progress(min(ratio, 1.0))
        elapsed_slot.metric("Elapsed", f"{elapsed:0.1f}s")
        eta_slot.metric("ETA", f"{remaining:0.1f}s")

    return _update


def _render_results(
    results: object,
    *,
    fold_selection: str | None,
) -> None:
    results_frame = None
    if hasattr(results, "results_frame"):
        candidate = getattr(results, "results_frame")
        if isinstance(candidate, pd.DataFrame):
            results_frame = candidate
    if isinstance(results, Mapping) and results_frame is None:
        candidate = results.get("results_frame")
        if isinstance(candidate, pd.DataFrame):
            results_frame = candidate
    if results_frame is None:
        st.warning("No results frame available to display.")
        return

    filtered_results = _filter_results_by_fold(results_frame, fold_selection)
    if filtered_results.empty:
        st.warning("No results available for the selected fold.")
        return

    from streamlit_app.components import mc_plots, mc_tables

    adapter_fold_selection = _fold_selection_for_adapters(fold_selection)
    summary = _cached_make_summary(results_frame, adapter_fold_selection)

    st.subheader("Summary")
    summary_table = mc_tables.render_summary_table(filtered_results)

    fold_id = None
    if fold_selection:
        tokens = fold_selection.split()
        if tokens and tokens[-1].isdigit():
            fold_id = int(tokens[-1])
    nav_paths = _extract_nav_paths(results, fold_id=fold_id)
    canonical_paths = (
        _cached_make_paths(nav_paths) if not nav_paths.empty else pd.DataFrame()
    )

    st.subheader("Charts")
    chart_bundle_inputs: dict[str, go.Figure] = {}
    tabs = st.tabs(["Core", "Diagnostics"])
    with tabs[0]:
        if nav_paths.empty:
            st.warning("No NAV paths available for the selected fold.")
        chart_bundle_inputs["Sharpe Histogram"] = mc_plots.render_sharpe_histogram(
            filtered_results
        )
        chart_bundle_inputs["Fan Chart"] = mc_plots.render_fan_chart(nav_paths)
        chart_bundle_inputs["Path Distribution"] = (
            mc_plots.render_path_distribution_chart(filtered_results)
        )
        chart_bundle_inputs["Risk Return"] = mc_plots.render_risk_return_chart(
            filtered_results
        )
        chart_bundle_inputs["Strategy Box Plot"] = mc_plots.render_box_plot(
            filtered_results
        )
        chart_bundle_inputs["Outcome CDF"] = mc_plots.render_cdf_plot(filtered_results)
    with tabs[1]:
        chart_bundle_inputs.update(_render_diagnostic_charts(summary, canonical_paths))

    st.subheader("Downloads")
    payloads = _build_download_payloads(summary_table, filtered_results)
    chart_bundle_payload, chart_bundle_warnings = _build_chart_bundle_payload(
        chart_bundle_inputs
    )
    if chart_bundle_payload is not None:
        payloads.append(chart_bundle_payload)
    warning_text = _png_export_warning_message(chart_bundle_warnings)
    if warning_text:
        st.warning(warning_text)
    for payload in payloads:
        st.download_button(**payload)


def _export_parquet_bytes(frame: pd.DataFrame) -> bytes | None:
    """Serialize ``frame`` to Parquet bytes in an engine-agnostic way.

    Returns ``None`` (instead of raising) when Parquet export is unavailable
    in the current environment, so callers can degrade gracefully to CSV-only.

    Why this exists: ``DataFrame.to_parquet(BytesIO())`` then ``getvalue()`` is
    NOT portable across Parquet engines. The offline WASM/Pyodide demo vendors
    ``fastparquet``, which CLOSES the ``BytesIO`` it is handed during the write,
    so the subsequent ``getvalue()`` raises ``ValueError: I/O operation on
    closed file`` and aborts the whole results render. ``pyarrow`` (used on CI)
    does not close the buffer, so the bug never surfaced locally — a classic
    cross-env failure. ``to_parquet(path=None)`` does not help either: pandas
    uses an internal ``BytesIO`` and calls ``getvalue()`` itself, hitting the
    same close. Writing to a real file PATH sidesteps it entirely: pandas (and
    the engine) own the file handle, and we read the bytes back afterwards.
    """

    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            frame.to_parquet(tmp_path, index=False)
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - any engine failure degrades to CSV-only
        return None


def _build_download_payloads(
    summary_table: pd.DataFrame,
    filtered_results: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return download button payloads for CSV, Parquet, and ZIP bundles.

    Parquet export is best-effort: if it fails (e.g. a buffer-closing engine or
    no Parquet engine at all), the Parquet download and the ZIP's Parquet entry
    are omitted but the CSV downloads still render, so the page never crashes.
    """

    summary_csv = summary_table.to_csv(index=False)
    path_frame = aggregate_monte_carlo_results(filtered_results).path_frame
    parquet_bytes = _export_parquet_bytes(path_frame)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payloads: list[dict[str, Any]] = [
        {
            "label": "Download summary CSV",
            "data": summary_csv,
            "file_name": f"mc_summary_{timestamp}.csv",
            "mime": "text/csv",
        }
    ]

    if parquet_bytes is not None:
        payloads.append(
            {
                "label": "Download representative paths (parquet)",
                "data": BytesIO(parquet_bytes),
                "file_name": f"mc_representative_paths_{timestamp}.parquet",
                "mime": "application/x-parquet",
            }
        )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("summary.csv", summary_csv)
        if parquet_bytes is not None:
            bundle.writestr("representative_paths.parquet", parquet_bytes)
    zip_buffer.seek(0)

    payloads.append(
        {
            "label": "Download ZIP bundle",
            "data": zip_buffer,
            "file_name": f"mc_bundle_{timestamp}.zip",
            "mime": "application/zip",
        }
    )

    return payloads


def _build_chart_bundle_payload(
    charts: Mapping[str, go.Figure],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not charts:
        return None, []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings: list[str] = []
    bundle_buffer = save_chart_bundle(charts, include_png=True, warnings=warnings)
    if not isinstance(bundle_buffer, BytesIO):
        with Path(bundle_buffer).open("rb") as handle:
            payload = BytesIO(handle.read())
        payload.seek(0)
    else:
        payload = bundle_buffer
        payload.seek(0)
    return (
        {
            "label": "Download charts bundle",
            "data": payload,
            "file_name": f"mc_charts_bundle_{timestamp}.zip",
            "mime": "application/zip",
        },
        warnings,
    )


def _png_export_warning_message(messages: list[str]) -> str | None:
    if not messages:
        return None
    return "Charts bundle PNG export warnings: " + " ".join(messages)


def render() -> None:
    """Render the Monte Carlo Simulation page."""

    st.title("Monte Carlo Simulation")
    st.write("Select a Monte Carlo scenario to explore and run simulations.")

    if MC_RUNNING_KEY not in st.session_state:
        st.session_state[MC_RUNNING_KEY] = False
    if MC_CANCEL_KEY not in st.session_state:
        st.session_state[MC_CANCEL_KEY] = False

    try:
        all_scenarios = list_scenarios()
    except Exception as exc:  # pragma: no cover - defensive guard
        st.error("Unable to load Monte Carlo scenarios.")
        with st.expander("Details"):
            st.write(str(exc))
        return

    available_tags = _collect_tags(all_scenarios)
    selected_tags = st.multiselect(
        "Filter by tags",
        options=available_tags,
        default=[],
        help="Show scenarios that match any of the selected tags.",
    )

    if selected_tags:
        try:
            scenarios = list_scenarios(tags=selected_tags)
        except Exception as exc:  # pragma: no cover - defensive guard
            st.error("Unable to apply tag filter.")
            with st.expander("Details"):
                st.write(str(exc))
            scenarios = []
    else:
        scenarios = all_scenarios

    if not scenarios:
        st.warning("No scenarios available for the selected filters.")
        return

    options = [entry.name for entry in scenarios]
    entry_map = _scenario_lookup(scenarios)
    selected_name = st.selectbox(
        "Scenario",
        options=options,
        index=0,
        format_func=lambda name: (
            f"{name} - {entry_map[name].description}"
            if name in entry_map and entry_map[name].description
            else name
        ),
    )
    selected_entry = entry_map.get(selected_name)

    if selected_entry and selected_entry.description:
        st.caption(selected_entry.description)

    if selected_entry and selected_entry.tags:
        st.write("Tags: " + ", ".join(selected_entry.tags))

    scenario: MonteCarloScenario | None = None
    if selected_entry:
        try:
            scenario = load_scenario(selected_entry.name)
        except Exception as exc:
            st.error("Unable to load the selected scenario.")
            with st.expander("Details"):
                st.write(str(exc))
            return

    if scenario is None:
        return

    settings = scenario.monte_carlo
    if not isinstance(settings, MonteCarloSettings):
        st.error("Scenario settings are not resolved.")
        return
    runner_kwargs, session_error = _session_runner_kwargs()
    if session_error:
        st.warning("Current Data/Model state could not be applied to Monte Carlo.")
        with st.expander("Details"):
            st.write(session_error)

    st.subheader("Run Overrides")
    override_cols = st.columns(2)
    with override_cols[0]:
        default_paths = _clamp_int(settings.n_paths, 100, 5000)
        n_paths = st.slider(
            "Number of paths",
            min_value=100,
            max_value=5000,
            value=default_paths,
            step=100,
        )
        default_horizon = _clamp_int(int(round(settings.horizon_years)), 5, 50)
        horizon_years = st.slider(
            "Horizon (years)",
            min_value=5,
            max_value=50,
            value=default_horizon,
            step=1,
        )
    with override_cols[1]:
        seed_value = "" if settings.seed is None else str(settings.seed)
        seed_text = st.text_input(
            "Random seed",
            value=seed_value,
            help="Leave empty for non-deterministic runs.",
        )
        jobs_default = _clamp_int(settings.jobs, 1, 16)
        jobs = st.slider(
            "Parallel jobs",
            min_value=1,
            max_value=16,
            value=jobs_default,
            step=1,
        )

    fold_selection = None
    fold_options = _fold_options(scenario)
    if fold_options:
        fold_selection = st.selectbox("Fold selection", options=fold_options, index=0)

    errors: list[str] = []
    if not 100 <= n_paths <= 5000:
        errors.append("Number of paths must be between 100 and 5000.")
    if not 5 <= horizon_years <= 50:
        errors.append("Horizon years must be between 5 and 50.")
    if not 1 <= jobs <= 16:
        errors.append("Parallel jobs must be between 1 and 16.")

    seed, seed_error = _coerce_seed(seed_text)
    if seed_error:
        errors.append(seed_error)

    for message in errors:
        st.error(message)

    button_cols = st.columns(2)
    run_clicked = False
    validate_clicked = False
    if not st.session_state.get(MC_RUNNING_KEY):
        with button_cols[0]:
            run_clicked = st.button(
                "Run simulation",
                type="primary",
                disabled=bool(errors),
            )
        with button_cols[1]:
            validate_clicked = st.button(
                "Validate scenario",
                disabled=bool(errors),
            )
    else:
        with button_cols[0]:
            if st.button("Cancel run", type="secondary"):
                st.session_state[MC_CANCEL_KEY] = True

    if validate_clicked:
        runner = MonteCarloRunner(scenario, **runner_kwargs)
        if hasattr(runner, "validate") and callable(getattr(runner, "validate")):
            try:
                issues = runner.validate()
                st.session_state[MC_LAST_VALIDATION_KEY] = issues
                if issues:
                    st.warning("Validation completed with issues.")
                    for issue in issues:
                        st.info(str(issue))
                else:
                    st.success("Validation completed successfully.")
            except Exception as exc:
                st.error("Validation failed.")
                with st.expander("Details"):
                    st.write(str(exc))
        else:
            st.info("Validation entrypoint is not available for this runner.")

    if run_clicked:
        st.session_state[MC_RUNNING_KEY] = True
        st.session_state[MC_CANCEL_KEY] = False
        st.session_state[MC_LAST_ERROR_KEY] = None
        progress_slot = st.empty()
        elapsed_slot = st.empty()
        eta_slot = st.empty()
        progress_bar = progress_slot.progress(0.0, text="Starting simulation...")
        start_time = monotonic()
        try:
            run_scenario = _build_override_scenario(
                scenario,
                n_paths=n_paths,
                horizon_years=horizon_years,
                seed=seed,
                jobs=jobs,
            )
            runner = MonteCarloRunner(run_scenario, **runner_kwargs)
            progress_callback = _progress_callback_factory(
                progress_bar=progress_bar,
                elapsed_slot=elapsed_slot,
                eta_slot=eta_slot,
                start_time=start_time,
            )
            results = runner.run(progress_callback=progress_callback, jobs=jobs)
            progress_bar.progress(1.0, text="Simulation complete.")
            st.session_state[MC_RESULTS_KEY] = results
            st.success("Simulation completed.")
        except _RunCancelled:
            st.warning("Simulation cancelled.")
            st.session_state[MC_RESULTS_KEY] = None
        except Exception as exc:
            st.error("Simulation failed.")
            st.session_state[MC_LAST_ERROR_KEY] = str(exc)
            with st.expander("Details"):
                st.write(str(exc))
        finally:
            st.session_state[MC_RUNNING_KEY] = False

    results = st.session_state.get(MC_RESULTS_KEY)
    if results is not None:
        st.divider()
        _render_results(results, fold_selection=fold_selection)
