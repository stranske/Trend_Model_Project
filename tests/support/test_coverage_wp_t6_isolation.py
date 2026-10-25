"""Collecting the t6 tests must preserve the shared Streamlit test double."""

import subprocess
import sys
from pathlib import Path


def test_collecting_t6_preserves_shared_placeholder() -> None:
    root = Path(__file__).resolve().parents[2]
    check = """
import runpy
from tests.support.dummy_streamlit import DummyStreamlit, Placeholder

before = dict(vars(Placeholder))
namespace = runpy.run_path("tests/streamlit/test_coverage_wp_t6.py")
assert dict(vars(Placeholder)) == before, "t6 collection changed shared Placeholder"
assert type(DummyStreamlit().empty()) is Placeholder
stub = namespace["CustomDummyStreamlit"]()
placeholder = stub.empty()
placeholder.text("local text")
placeholder.error("local error")
placeholder.metric("local metric", 3)
assert stub.write_calls == ["local text"]
assert stub.error_messages == ["local error"]
assert stub.metric_calls == [("local metric", 3)]
assert placeholder.empty() is None
"""
    result = subprocess.run([sys.executable, "-c", check], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
