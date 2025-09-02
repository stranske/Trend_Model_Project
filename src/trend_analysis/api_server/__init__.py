"""Minimal HTTP API server for demo purposes.

Exposes a simple health check endpoint at ``/health``.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Tuple


class RequestHandler(BaseHTTPRequestHandler):
    """Handle basic GET requests."""

    def do_GET(self) -> None:  # noqa: N802 (Http server interface)
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Silence default logging in tests/demos
        return


def run(host: str = "127.0.0.1", port: int = 8080) -> Tuple[str, int]:
    """Run the demo API server and block until interrupted.

    Returns the host and port used.
    """
    server = HTTPServer((host, port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return host, port


if __name__ == "__main__":
    run()
