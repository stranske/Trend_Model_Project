FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt pyproject.toml ./

# Ensure build tools are available
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with retry logic for network issues
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt \
    && pip install --no-cache-dir pytest

# Copy source code and essential files
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY README.md LICENSE ./

# Install build dependencies and the package in development mode for CLI access
# Ensure build dependencies are available and handle editable install gracefully
RUN pip install --no-cache-dir setuptools>=61 wheel build && \
    (pip install --no-cache-dir -e .[app] && echo "Package installed successfully with [app] extra") || \
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

# Healthcheck for Streamlit app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command runs the main Streamlit app
CMD ["streamlit", "run", "src/trend_portfolio_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
