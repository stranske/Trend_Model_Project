from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from streamlit_app.components.data_cache import cache_key_for_frame


def _load_model_page():
    module_path = Path(__file__).resolve().parents[2] / "streamlit_app" / "pages" / "2_Model.py"
    spec = importlib.util.spec_from_file_location("streamlit_model_page", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_cache_key_includes_core_fields():
    module = _load_model_page()
    key = module._build_chain_cache_key(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.2,
    )
    assert key["provider"] == "openai"
    assert key["model"] == "gpt-4o-mini"
    assert key["temperature"] == 0.2


def test_cache_key_changes_detects_core_fields():
    module = _load_model_page()
    previous = module._build_chain_cache_key(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    current = module._build_chain_cache_key(
        provider="anthropic",
        model="claude-3-5-sonnet",
        temperature=0.4,
    )
    changed = module._cache_key_changes(
        previous,
        current,
        module._CONFIG_CHAIN_CORE_FIELDS,
    )
    assert "provider" in changed
    assert "model" in changed
    assert "temperature" in changed


def test_llm_cache_key_tracks_connection_fields():
    module = _load_model_page()
    key = module._build_llm_cache_key(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://example.test",
        organization="org",
        timeout=42.0,
        max_retries=3,
        extra_payload_hash="hash",
        api_key_fingerprint="fingerprint",
    )
    assert key["provider"] == "openai"
    assert key["model"] == "gpt-4o-mini"
    assert key["base_url"] == "https://example.test"
    assert key["organization"] == "org"
    assert key["timeout"] == 42.0
    assert key["max_retries"] == 3
    assert key["extra_payload_hash"] == "hash"
    assert key["api_key_fingerprint"] == "fingerprint"


def test_invalidation_fields_include_connection_overrides():
    module = _load_model_page()
    assert "base_url" in module._CONFIG_CHAIN_INVALIDATION_FIELDS
    assert "organization" in module._CONFIG_CHAIN_INVALIDATION_FIELDS


def test_dataframe_cache_key_is_digest_and_tracks_schema_index_and_data():
    frame = pd.DataFrame({"returns": [0.1, 0.2]}, index=pd.Index(["a", "b"], name="asset"))

    key = cache_key_for_frame(frame)

    assert len(key) == 64
    assert key == cache_key_for_frame(frame.copy())
    assert key != cache_key_for_frame(frame.rename(columns={"returns": "risk"}))
    assert key != cache_key_for_frame(frame.set_axis(["x", "y"], axis="index"))
    assert key != cache_key_for_frame(frame.assign(returns=[0.1, 0.3]))
