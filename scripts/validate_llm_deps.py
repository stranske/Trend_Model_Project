"""Verify that LLM extras resolve compatible Python and Pydantic versions."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
from typing import Iterable


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


def main() -> int:
    llm_packages = ("langchain", "langchain_core", "langchain_community")
    present = _find_first_installed(llm_packages)
    if present is None:
        print("LLM extras not installed; skipping compatibility checks.")
        return 0

    if sys.version_info < (3, 10):
        print(
            "Python >=3.10 is required for LLM extras. "
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

    print(
        "LLM dependency checks passed: "
        f"Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"Pydantic {pydantic_version}.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
