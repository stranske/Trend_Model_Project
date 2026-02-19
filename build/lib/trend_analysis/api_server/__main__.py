"""Entry point for running the FastAPI server."""

from . import run

if __name__ == "__main__":
    # Run FastAPI server on port 8000 to match docker-compose.yml
    run(host="0.0.0.0", port=8000)
