"""Command-line interface for the LLM proxy server."""

from __future__ import annotations

import argparse
import logging
import sys

from ..logging_setup import setup_logging
from .server import run_proxy


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible LLM proxy")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=(
            "Host to bind the proxy server (default: 0.0.0.0). "
            "Use 127.0.0.1 for local-only access."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8799,
        help="Port to bind the proxy server (default: 8799)",
    )
    parser.add_argument(
        "--upstream",
        default=None,
        help="Override upstream OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    announce_logs = __name__ == "__main__"
    log_path = setup_logging(
        level=args.log_level,
        app_name="llm_proxy",
        enable_console=announce_logs,
    )
    logger = logging.getLogger(__name__)
    logger.info("LLM proxy logs stored at %s", log_path)
    if announce_logs:
        print(f"LLM proxy logs stored at {log_path}", file=sys.stderr)

    try:
        run_proxy(upstream_base=args.upstream, host=args.host, port=args.port)
        return 0
    except KeyboardInterrupt:
        print("\nLLM proxy server stopped by user")
        return 0
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Error starting LLM proxy: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
