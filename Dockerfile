FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt pyproject.toml ./

# Ensure build tools are available and pinned for reproducibility
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir setuptools==69.5.1 wheel==0.43.0

# Install Python dependencies with retry logic for network issues
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt \
    && pip install --no-cache-dir pytest

# Copy the rest of the project including tests and docs
COPY . .

# Install build dependencies and the package in development mode for CLI access
# Ensure build dependencies are available and handle editable install gracefully
RUN pip install --no-cache-dir setuptools>=61 wheel build && \
    (pip install --no-cache-dir -v -e .[app] && echo "Package installed successfully with [app] extra") || \
    (echo "Warning: Editable install with [app] extra failed. Falling back to individual package install." && \
     pip install --no-cache-dir streamlit>=1.30 streamlit-sortables && \
     pip install --no-cache-dir -e . && echo "Package installed in editable mode without [app] extra")

# Create demo data directory and ensure proper permissions
RUN mkdir -p demo && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck using new /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/health || exit 1

# Default command runs the health wrapper (which starts Streamlit internally)
ENV PYTHONPATH="/app/src"
ENV STREAMLIT_APP_PATH="/app/src/trend_portfolio_app/app.py"
ENV HEALTH_WRAPPER_PORT="8501"
ENV STREAMLIT_PORT="8502"
ENV STREAMLIT_SERVER_HEADLESS="true"
CMD ["python", "-m", "trend_portfolio_app.health_wrapper"]
