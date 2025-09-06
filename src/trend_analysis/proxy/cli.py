"""Command-line interface for the Streamlit proxy server."""

import argparse
import logging
import sys

from .server import run_proxy


def main() -> int:
    """Main entry point for the proxy CLI."""
    parser = argparse.ArgumentParser(
        description="WebSocket-capable proxy for Streamlit applications"
    )
    
    parser.add_argument(
        "--streamlit-host",
        default="localhost", 
        help="Host where Streamlit is running (default: localhost)"
    )
    parser.add_argument(
        "--streamlit-port", 
        type=int,
        default=8501,
        help="Port where Streamlit is running (default: 8501)"
    )
    parser.add_argument(
        "--proxy-host",
        default="0.0.0.0",
        help="Host to bind the proxy server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--proxy-port",
        type=int, 
        default=8500,
        help="Port to bind the proxy server (default: 8500)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        run_proxy(
            streamlit_host=args.streamlit_host,
            streamlit_port=args.streamlit_port,
            proxy_host=args.proxy_host,
            proxy_port=args.proxy_port
        )
        return 0
    except KeyboardInterrupt:
        print("\nProxy server stopped by user")
        return 0
    except Exception as e:
        print(f"Error starting proxy: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())