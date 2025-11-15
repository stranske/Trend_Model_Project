FROM python:3.11-slim@sha256:316d89b74c4d467565864be703299878ca7a97893ed44ae45f6acba5af09d154

# Deterministic hash seed (can be overridden at build time)
ARG PY_HASH_SEED=0
ENV PYTHONHASHSEED=${PY_HASH_SEED}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy project metadata and lock file first for better Docker layer caching
COPY pyproject.toml requirements.lock ./

# Ensure build tools are available and pinned for reproducibility
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir setuptools==69.5.1 wheel==0.43.0 uv

# Install Python dependencies from the pinned lock file
RUN uv pip sync --system requirements.lock

# Copy the rest of the project including tests and docs
COPY . .

# Install the package in editable mode without re-resolving dependencies
RUN pip install --no-cache-dir --no-deps -e .[app]

# Create demo data directory and ensure proper permissions
RUN mkdir -p demo && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose health wrapper port
EXPOSE 8000

# Healthcheck using the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Default command runs the health wrapper (which starts Streamlit internally)
ENV PYTHONPATH="/app/src"
ENV STREAMLIT_APP_PATH="/app/src/trend_portfolio_app/app.py"
ENV HEALTH_PORT="8000"
ENV STREAMLIT_PORT="8502"
ENV STREAMLIT_SERVER_HEADLESS="true"
CMD ["python", "-m", "trend_portfolio_app.health_wrapper"]
