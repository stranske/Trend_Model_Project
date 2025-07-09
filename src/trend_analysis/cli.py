from __future__ import annotations

import argparse
from .config import load


def main(argv: list[str] | None = None) -> int:
    """Simple command-line interface for loading configuration."""
    parser = argparse.ArgumentParser(prog="trend-analysis")
    parser.add_argument("-c", "--config", help="Path to YAML config")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    if args.version:
        print(cfg.version)
    else:
        print(cfg.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
