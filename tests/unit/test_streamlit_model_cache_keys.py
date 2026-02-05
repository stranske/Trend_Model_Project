from __future__ import annotations

import importlib.util
from pathlib import Path


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
        base_url="https://example.test",
        organization="org",
        temperature=0.2,
    )
    assert key["provider"] == "openai"
    assert key["model"] == "gpt-4o-mini"
    assert key["base_url"] == "https://example.test"
    assert key["organization"] == "org"
    assert key["temperature"] == 0.2


def test_cache_key_changes_detects_core_fields():
    module = _load_model_page()
    previous = module._build_chain_cache_key(
        provider="openai",
        model="gpt-4o-mini",
        base_url=None,
        organization=None,
        temperature=0.0,
    )
    current = module._build_chain_cache_key(
        provider="anthropic",
        model="claude-3-5-sonnet",
        base_url="https://example.test",
        organization="org",
        temperature=0.4,
    )
    changed = module._cache_key_changes(
        previous,
        current,
        module._CONFIG_CHAIN_CORE_FIELDS,
    )
    assert "provider" in changed
    assert "model" in changed
    assert "base_url" in changed
    assert "organization" in changed
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
