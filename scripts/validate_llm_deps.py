"""Verify that LLM extras resolve compatible Python and Pydantic versions."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterable

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def _find_first_installed(packages: Iterable[str]) -> str | None:
    for package in packages:
        if importlib.util.find_spec(package) is not None:
            return package
    return None


def _parse_major(version: str) -> int:
    match = re.match(r"(\d+)", version)
    if not match:
        raise ValueError(f"Unable to parse major version from '{version}'.")
    return int(match.group(1))


def _parse_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Unable to parse major/minor version from '{version}'.")
    return int(match.group(1)), int(match.group(2))


def _declared_llm_ranges() -> dict[str, SpecifierSet]:
    """Load supported LangChain ranges from the project metadata."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        requirements = tomllib.load(handle)["project"]["optional-dependencies"]["llm"]

    ranges: dict[str, SpecifierSet] = {}
    for requirement in requirements:
        name = re.split(r"[<>=!~\[]", requirement, maxsplit=1)[0].strip()
        if name in {"langchain", "langchain-core", "langchain-community"}:
            ranges[name] = SpecifierSet(requirement.removeprefix(name))
    return ranges


def main() -> int:
    llm_packages = ("langchain", "langchain_core", "langchain_community")
    langchain_distributions = {
        "langchain": "langchain",
        "langchain_core": "langchain-core",
        "langchain_community": "langchain-community",
    }
    declared_ranges = _declared_llm_ranges()
    present = _find_first_installed(llm_packages)
    if present is None:
        print("LLM extras not installed; skipping compatibility checks.")
        return 0

    if sys.version_info < (3, 12):
        print(
            "Python >=3.12 is required for LLM extras. "
            f"Detected {sys.version_info.major}.{sys.version_info.minor}.",
            file=sys.stderr,
        )
        return 1

    try:
        pydantic_version = importlib.metadata.version("pydantic")
    except importlib.metadata.PackageNotFoundError:
        print("Pydantic is required when LLM extras are installed.", file=sys.stderr)
        return 1

    try:
        pydantic_major = _parse_major(pydantic_version)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if pydantic_major < 2:
        print(
            "Pydantic v2 is required for LLM extras. " f"Detected {pydantic_version}.",
            file=sys.stderr,
        )
        return 1

    missing_langchain = []
    incompatible_langchain = []
    for distribution in langchain_distributions.values():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing_langchain.append(distribution)
            continue

        expected = declared_ranges.get(distribution)
        if expected is None:
            incompatible_langchain.append(
                f"{distribution}=={version} (no expected version range configured)"
            )
            continue

        try:
            compatible = Version(version) in expected
        except ValueError:
            compatible = False
        if not compatible:
            incompatible_langchain.append(f"{distribution}=={version} (expected {expected})")

    if missing_langchain:
        print(
            "LLM extras require langchain, langchain-core, and langchain-community. "
            f"Missing: {', '.join(missing_langchain)}.",
            file=sys.stderr,
        )
        return 1

    if incompatible_langchain:
        print(
            "LangChain packages must be within the pyproject.toml declared ranges. "
            f"Detected {', '.join(incompatible_langchain)}.",
            file=sys.stderr,
        )
        return 1

    print(
        "LLM dependency checks passed: "
        f"Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"Pydantic {pydantic_version}.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
