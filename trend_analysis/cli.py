from __future__ import annotations

import argparse

from .config import load
from .pipeline import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trend analysis CLI")
    parser.add_argument("config", nargs="?", help="Path to config YAML")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    df = run(cfg)
    print(df.head())


__all__ = ["main"]
