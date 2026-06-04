"""Entry point for running the FastAPI server."""

from . import run

if __name__ == "__main__":
    # Run FastAPI server on port 8000 to match docker-compose.yml
    run(host="127.0.0.1", port=8000)
