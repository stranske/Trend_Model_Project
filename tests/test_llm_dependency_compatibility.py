"""Compatibility checks for LLM extras."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
import tomllib
import warnings
from pathlib import Path

import pydantic
import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def _parse_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        pytest.fail(f"Unable to parse major/minor version from '{version}'.")
    return int(match.group(1)), int(match.group(2))


def _get_declared_version_range(package: str) -> SpecifierSet:
    """Extract declared version range from pyproject.toml."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    # Search in optional-dependencies llm group
    llm_deps = data.get("project", {}).get("optional-dependencies", {}).get("llm", [])

    for dep in llm_deps:
        # Extract package name
        pkg_name = dep.split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip()
        if pkg_name == package:
            # Extract specifier (e.g., ">=1.2,<1.3")
            spec_str = dep.replace(pkg_name, "", 1).strip()
            if spec_str:
                return SpecifierSet(spec_str)

    return SpecifierSet()


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


@pytest.mark.parametrize(
    "distribution",
    ["langchain", "langchain-core", "langchain-community"],
)
def test_langchain_versions_match_pyproject(distribution: str) -> None:
    """Validate installed langchain versions match pyproject.toml constraints.

    This test ensures the installed versions are within the ranges specified
    in pyproject.toml [project.optional-dependencies] llm group, preventing
    drift between declared and actual dependencies.

    ✅ This test dynamically reads version ranges from pyproject.toml,
       so it won't break when dependencies are updated.
    """
    pytest.importorskip("langchain")

    installed_version = Version(importlib.metadata.version(distribution))
    declared_range = _get_declared_version_range(distribution)

    if not declared_range:
        pytest.fail(
            f"{distribution} not found in pyproject.toml [project.optional-dependencies] llm group"
        )

    assert installed_version in declared_range, (
        f"{distribution} version {installed_version} not in declared range {declared_range}. "
        f"Check pyproject.toml and requirements.lock for consistency."
    )
