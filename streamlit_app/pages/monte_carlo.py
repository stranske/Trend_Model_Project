"""Monte Carlo simulation page for the Streamlit application."""

from __future__ import annotations

import copy
from datetime import datetime
from io import BytesIO
from time import monotonic
from typing import Iterable, Mapping
import zipfile

import pandas as pd
import streamlit as st

from streamlit_app.components import mc_plots, mc_tables
from streamlit_app.components.progress_eta import progress_ratio_and_remaining
from trend_analysis.monte_carlo.aggregator import aggregate_monte_carlo_results
from trend_analysis.monte_carlo.registry import (
    ScenarioRegistryEntry,
    list_scenarios,
    load_scenario,
)
from trend_analysis.monte_carlo.runner import MonteCarloRunner
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings


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


class _RunCancelled(RuntimeError):
    """Raised when the user cancels a Monte Carlo run."""


def _collect_tags(entries: Iterable[ScenarioRegistryEntry]) -> list[str]:
    tags: set[str] = set()
    for entry in entries:
        tags.update(entry.tags)
    return sorted(tags)


def _scenario_lookup(entries: Iterable[ScenarioRegistryEntry]) -> dict[str, ScenarioRegistryEntry]:
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


def _filter_results_by_fold(results: pd.DataFrame, selection: str | None) -> pd.DataFrame:
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


def _extract_nav_paths(results: object) -> pd.DataFrame:
    if hasattr(results, "nav_paths"):
        nav_paths = getattr(results, "nav_paths")
        if isinstance(nav_paths, pd.DataFrame):
            return nav_paths
    if hasattr(results, "metadata"):
        metadata = getattr(results, "metadata")
        if isinstance(metadata, Mapping):
            nav_paths = metadata.get("nav_paths")
            if isinstance(nav_paths, pd.DataFrame):
                return nav_paths
    if isinstance(results, Mapping):
        nav_paths = results.get("nav_paths")
        if isinstance(nav_paths, pd.DataFrame):
            return nav_paths
    return pd.DataFrame()


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

    st.subheader("Summary")
    summary_table = mc_tables.render_summary_table(filtered_results)

    st.subheader("Charts")
    nav_paths = _extract_nav_paths(results)
    mc_plots.render_sharpe_histogram(filtered_results)
    mc_plots.render_fan_chart(nav_paths)
    mc_plots.render_box_plot(filtered_results)
    mc_plots.render_cdf_plot(filtered_results)

    st.subheader("Downloads")
    summary_csv = summary_table.to_csv(index=False)
    st.download_button(
        label="Download summary CSV",
        data=summary_csv,
        file_name=f"mc_summary_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )

    path_frame = aggregate_monte_carlo_results(filtered_results).path_frame
    parquet_buffer = BytesIO()
    path_frame.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    st.download_button(
        label="Download representative paths (parquet)",
        data=parquet_buffer,
        file_name=f"mc_representative_paths_{datetime.now():%Y%m%d_%H%M%S}.parquet",
        mime="application/x-parquet",
    )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("summary.csv", summary_csv)
        bundle.writestr("representative_paths.parquet", parquet_buffer.getvalue())
    zip_buffer.seek(0)
    st.download_button(
        label="Download ZIP bundle",
        data=zip_buffer,
        file_name=f"mc_bundle_{datetime.now():%Y%m%d_%H%M%S}.zip",
        mime="application/zip",
    )


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
    selected_name = st.selectbox("Scenario", options=options, index=0)
    entry_map = _scenario_lookup(scenarios)
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
        runner = MonteCarloRunner(scenario)
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
            runner = MonteCarloRunner(run_scenario)
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


if _should_auto_render():
    render()
