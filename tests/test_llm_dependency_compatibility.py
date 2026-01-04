"""Compatibility checks for LLM extras."""

from __future__ import annotations

import importlib
import sys
import warnings

import pydantic
import pytest


def test_llm_extras_require_python_310_plus() -> None:
    assert sys.version_info >= (3, 10)


def test_llm_extras_use_pydantic_v2() -> None:
    major = int(pydantic.__version__.split(".", 1)[0])
    assert major == 2


def test_langchain_import_has_no_pydantic_warnings() -> None:
    pytest.importorskip("langchain")
    sys.modules.pop("langchain", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("langchain")
    assert not any("pydantic" in str(warning.message).lower() for warning in caught)
