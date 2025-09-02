"""Minimal HTTP API server for demo purposes.

Exposes a simple health check endpoint at ``/health``.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer


class RequestHandler(BaseHTTPRequestHandler):
    """Handle basic GET requests."""

    def do_GET(self):  # noqa: N802 (Http server interface)
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the minimal HTTP server."""
    server = HTTPServer((host, port), RequestHandler)
    server.serve_forever()
