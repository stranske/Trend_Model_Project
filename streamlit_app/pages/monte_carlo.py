"""Monte Carlo simulation page for the Streamlit application."""

from __future__ import annotations

from typing import Iterable

import streamlit as st

from trend_analysis.monte_carlo.registry import ScenarioRegistryEntry, list_scenarios


def _should_auto_render() -> bool:
    """Return True when running inside an active Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _collect_tags(entries: Iterable[ScenarioRegistryEntry]) -> list[str]:
    tags: set[str] = set()
    for entry in entries:
        tags.update(entry.tags)
    return sorted(tags)


def _scenario_lookup(entries: Iterable[ScenarioRegistryEntry]) -> dict[str, ScenarioRegistryEntry]:
    return {entry.name: entry for entry in entries}


def render() -> None:
    """Render the Monte Carlo Simulation page."""

    st.title("Monte Carlo Simulation")
    st.write("Select a Monte Carlo scenario to explore and run simulations.")

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


if _should_auto_render():
    render()
