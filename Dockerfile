ARG PYTHON_IMAGE=python:3.12.2-slim

FROM ${PYTHON_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Reusable deterministic hash seed (runtime stage reuses the same ARG)
ARG PY_HASH_SEED=0
ENV PYTHONHASHSEED=${PY_HASH_SEED}

# Tooling needed to build wheels and assets
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl git nodejs npm && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build

# Copy project metadata first for caching
COPY pyproject.toml requirements.lock ./

# Install deterministic build tooling plus uv for lock-friendly syncs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools==69.5.1 wheel==0.43.0 uv

# Install locked dependencies inside the virtual environment
RUN uv pip sync requirements.lock

# Copy the remainder of the repo
COPY . .

# Install the package in editable mode without re-resolving dependencies
RUN pip install --no-cache-dir --no-deps -e .[app]

FROM ${PYTHON_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PY_HASH_SEED=0
ENV PYTHONHASHSEED=${PY_HASH_SEED}

# Runtime deps only (builder already handled compilation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder image
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

# Copy project sources (already filtered by .dockerignore)
COPY --from=builder /build /app

# Create non-root user after sources exist so chown can include everything
RUN useradd -m -u 1001 appuser

# Verify the runtime environment matches the lock file exactly (after normalizing
# package name casing and hyphen/underscore differences).
RUN set -eux && \
    pip freeze --exclude-editable | \
        grep -v '^trend-model @' | \
        grep -vE '^(pip|setuptools|wheel|uv)==' \
        > /tmp/runtime-freeze.raw && \
    cp requirements.lock /tmp/lock.raw && \
    python - <<'PY'
import difflib
from pathlib import Path

def canonicalize(path: str) -> list[str]:
    rows: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        norm_name = name.lower().replace("_", "-")
        rows.append(f"{norm_name}=={version.strip()}\n")
    return sorted(rows)

lock_rows = canonicalize("/tmp/lock.raw")
runtime_rows = canonicalize("/tmp/runtime-freeze.raw")
diff = list(
    difflib.unified_diff(
        lock_rows,
        runtime_rows,
        fromfile="requirements.lock",
        tofile="pip-freeze",
    )
)
if diff:
    raise SystemExit("".join(diff))
PY

# Ensure demo outputs can be written at runtime and drop permissions
RUN mkdir -p demo && \
    chown -R appuser:appuser /app /opt/venv

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
