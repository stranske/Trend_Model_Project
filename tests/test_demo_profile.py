"""Smoke tests for the stlite browser-demo profiles and build (issue #5343)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from streamlit_app import demo_profile as dp

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Pure profile resolution / gating logic
# ---------------------------------------------------------------------------
def test_default_is_presentation_safe() -> None:
    assert dp.DEFAULT_PROFILE == dp.PRESENTATION_SAFE
    assert dp.VALID_PROFILES == (dp.PRESENTATION_SAFE, dp.PUBLIC_LLM_DEMO)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, dp.PRESENTATION_SAFE),
        ("", dp.PRESENTATION_SAFE),
        ("  ", dp.PRESENTATION_SAFE),
        ("nonsense", dp.PRESENTATION_SAFE),
        ("presentation_safe", dp.PRESENTATION_SAFE),
        ("  Public_LLM_Demo  ", dp.PUBLIC_LLM_DEMO),
        ("PUBLIC_LLM_DEMO", dp.PUBLIC_LLM_DEMO),
    ],
)
def test_normalize_profile(value: Any, expected: str) -> None:
    assert dp.normalize_profile(value) == expected


def test_resolve_profile_precedence() -> None:
    # session beats query beats env.
    assert (
        dp.resolve_profile(
            session="public_llm_demo", query_param="presentation_safe", env="presentation_safe"
        )
        == dp.PUBLIC_LLM_DEMO
    )
    assert (
        dp.resolve_profile(session=None, query_param="public_llm_demo", env="presentation_safe")
        == dp.PUBLIC_LLM_DEMO
    )
    assert (
        dp.resolve_profile(session=None, query_param=None, env="public_llm_demo")
        == dp.PUBLIC_LLM_DEMO
    )
    assert dp.resolve_profile() == dp.PRESENTATION_SAFE


def test_resolve_profile_ignores_invalid_and_falls_through() -> None:
    # An invalid higher-precedence signal must not disable safety; fall through.
    assert (
        dp.resolve_profile(session="bogus", query_param="public_llm_demo", env=None)
        == dp.PUBLIC_LLM_DEMO
    )
    assert (
        dp.resolve_profile(session="bogus", query_param="", env="also-bogus")
        == dp.PRESENTATION_SAFE
    )


def test_gating_helpers_presentation_safe() -> None:
    p = dp.PRESENTATION_SAFE
    assert dp.llm_enabled(p) is False
    assert dp.custom_analysis_enabled(p) is False
    assert dp.uploads_enabled(p) is False


def test_gating_helpers_public_llm_demo() -> None:
    p = dp.PUBLIC_LLM_DEMO
    assert dp.llm_enabled(p) is True
    assert dp.custom_analysis_enabled(p) is True
    assert dp.uploads_enabled(p) is True


def test_profile_label_known_and_default() -> None:
    assert "Presentation-safe" in dp.profile_label(dp.PRESENTATION_SAFE)
    assert "LangChain" in dp.profile_label(dp.PUBLIC_LLM_DEMO)
    # Unknown normalises to the safe default label.
    assert dp.profile_label("bogus") == dp.profile_label(dp.PRESENTATION_SAFE)


# ---------------------------------------------------------------------------
# Sidebar switcher render (with a fake Streamlit module)
# ---------------------------------------------------------------------------
class _FakeSidebar:
    def __init__(self, selected: str) -> None:
        self._selected = selected
        self.captions: list[str] = []

    def selectbox(self, _label: str, options, index: int, **_kw):  # noqa: ANN001
        return self._selected

    def caption(self, text: str) -> None:
        self.captions.append(text)


class _FakeStreamlit:
    def __init__(self, selected: str) -> None:
        self.sidebar = _FakeSidebar(selected)
        self.session_state: dict[str, Any] = {}
        self.query_params: dict[str, Any] = {}


def test_render_profile_controls_switch_to_llm() -> None:
    fake = _FakeStreamlit(dp.PUBLIC_LLM_DEMO)
    active = dp.render_profile_controls(fake)
    assert active == dp.PUBLIC_LLM_DEMO
    assert fake.session_state[dp.PROFILE_SESSION_KEY] == dp.PUBLIC_LLM_DEMO
    assert any("LLM" in c for c in fake.sidebar.captions)


def test_render_profile_controls_presentation_safe_caption() -> None:
    fake = _FakeStreamlit(dp.PRESENTATION_SAFE)
    active = dp.render_profile_controls(fake)
    assert active == dp.PRESENTATION_SAFE
    assert any("Presentation-safe" in c for c in fake.sidebar.captions)


# ---------------------------------------------------------------------------
# Browser-demo artifact + build manifest
# ---------------------------------------------------------------------------
def _load_build_module():
    path = REPO_ROOT / "scripts" / "build_wasm_demo.py"
    spec = importlib.util.spec_from_file_location("build_wasm_demo", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wasm_artifacts_exist() -> None:
    wasm_dir = REPO_ROOT / "demo" / "wasm"
    index = (wasm_dir / "index.html").read_text(encoding="utf-8")
    assert "stlite" in index
    assert "manifest.json" in index
    # Default must be the safe profile.
    assert "presentation_safe" in index
    assert (wasm_dir / "README.md").is_file()


def test_manifest_builder_well_formed() -> None:
    # manifest.json is a generated build artifact (gitignored); validate the
    # builder output and that --check round-trips, rather than a committed file.
    build = _load_build_module()
    fresh = build.build_manifest(REPO_ROOT)

    assert fresh["entrypoint"] == "streamlit_app/app.py"
    assert fresh["default_profile"] == dp.PRESENTATION_SAFE
    # presentation_safe must not pull LangChain; public_llm_demo must.
    safe_reqs = " ".join(fresh["requirements"]["presentation_safe"]).lower()
    llm_reqs = " ".join(fresh["requirements"]["public_llm_demo"]).lower()
    assert "langchain" not in safe_reqs
    assert "langchain" in llm_reqs
    assert "langchain-openai" in llm_reqs
    assert "langchain-anthropic" in llm_reqs
    assert "langchain-ollama" in llm_reqs
    # The entrypoint and bundled synthetic data must be in the file list.
    assert "streamlit_app/app.py" in fresh["files"]
    assert "streamlit_app/demo_profile.py" in fresh["files"]
    assert "src/trend/__init__.py" in fresh["files"]
    assert "src/utils/__init__.py" in fresh["files"]
    assert "demo/demo_returns.csv" in fresh["files"]

    # The manifest is JSON-serialisable and stable across a round-trip.
    assert json.loads(json.dumps(fresh)) == fresh
