"""Entrypoint for running the LLM proxy module."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
