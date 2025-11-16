# Docker Quickstart Guide

This guide walks through building and running the new multi-stage (Python 3.12.2-slim)
container image. The builder/runtime split keeps dependencies reproducible and
enforces a lock-file parity check every time you build.

## Quick Start

### Option 1: Build + Run Locally (Recommended)

Use Docker Compose to build the image once and launch the Streamlit app:

```bash
docker compose up --build app
```

The first invocation builds the multi-stage image (tagged `trend-model:local`).
Subsequent runs reuse the cached layers, so `docker compose up app` is usually
enough. Open http://localhost:8501 after the container reports that Streamlit
started.

### Option 2: Pull the Published Image

Need a one-off run without cloning the repo? Use the published artifact:

```bash
docker run -p 8501:8501 ghcr.io/stranske/trend-model:latest
```

This mirrors the Compose workflow but skips the local build step.

### Option 3: Use the CLI

Run analysis directly from command line:

```bash
# Generate demo data and run analysis
docker run --rm trend-model:local python scripts/generate_demo.py

# Run trend analysis with demo config
docker run --rm -v $(pwd):/workspace trend-model:local \
  trend-analysis -c config/demo.yml
```

When using the published image, replace `trend-model:local` with
`ghcr.io/stranske/trend-model:latest`.

### Option 4: Interactive Shell

Get a shell inside the container for development:

```bash
docker run -it --rm trend-model:local bash
```

### Option 5: Docker Compose (App + API)

Bring up the Streamlit app and a minimal API server with one command:

```bash
docker compose up --build api
```

The compose file builds the local image (if needed) and mounts `./data` into
`/app/data` for both services.

## Working with Your Data

To analyze your own data, mount a local directory containing your CSV files:

```bash
docker run -p 8501:8501 -v /path/to/your/data:/app/data trend-model:local
```

Your data will be accessible at `/app/data/` inside the container. Swap
`trend-model:local` with `ghcr.io/stranske/trend-model:latest` if you are
running the published image.

## Dependency parity check

Every build now runs `pip freeze --exclude-editable` inside the final image and
diffs the output against `requirements.lock`. The build fails if the lock file
and installed wheels ever diverge, so remember to re-run `uv pip compile ...`
whenever you modify `pyproject.toml`.

## Container Features

### Included Components
- **Streamlit Web App**: Full-featured interactive interface at port 8501
- **CLI Tools**: `trend-analysis`, `trend-multi-analysis`, and `trend-model` commands
- **Demo Data**: Built-in demo datasets for testing
- **All Dependencies**: Pandas, NumPy, PyYAML, and all required packages

### Security Features
- **Non-root User**: Runs as user `appuser` (UID 1001) for security
- **Minimal Base**: Multi-stage Python 3.12.2-slim image copies only the
  prepared virtualenv into runtime, keeping attack surface small
- **Dependency Parity**: Build fails if `pip freeze` differs from
  `requirements.lock`, guaranteeing runtime reproducibility
- **Health Monitoring**: Built-in healthcheck endpoint at `/_stcore/health`

### Available CLI Commands

Inside the container, you can use:

```bash
# Main analysis command
trend-analysis -c config/demo.yml

# Multi-period analysis
trend-multi-analysis -c config/demo.yml

# General trend model CLI
trend-model --help

# Direct Python execution
python -m trend_analysis.run_analysis -c config/demo.yml
```

## Development Workflow

### Local Development with Docker

1. **Build locally** (after cloning the repo):
  ```bash
  git clone https://github.com/stranske/Trend_Model_Project.git
  cd Trend_Model_Project
  docker compose build
  ```
  The compose file tags the result as `trend-model:local`. If you changed
  dependencies, regenerate `requirements.lock` first so the build-time parity
  check passes.

2. **Run with live code updates**:
  ```bash
  docker run -p 8501:8501 -v $(pwd):/app trend-model:local
  ```

### Testing the Container

Verify the container works correctly:

```bash
# Test web interface starts correctly
docker run -d --name trend-test -p 8501:8501 trend-model:local
sleep 10
curl -f http://localhost:8501/_stcore/health
docker stop trend-test && docker rm trend-test

# Test CLI functionality
docker run --rm trend-model:local trend-analysis --help

# Comprehensive test suite
./test_docker.sh
```

Swap in `ghcr.io/stranske/trend-model:latest` if you want to validate the
published image. The Dockerfile now installs dependencies inside a virtualenv
via `uv pip sync requirements.lock` followed by `pip install --no-deps -e .[app]`
and fails the build if anything deviates from the lock file, so keeping the lock
current remains essential.

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Use a different port
docker run -p 8502:8501 trend-model:local
# Then visit http://localhost:8502
```

**Permission denied for mounted volumes:**
```bash
# Ensure your local directory has appropriate permissions
chmod 755 /path/to/your/data
```

**Container fails to start:**
```bash
# Check container logs
docker logs <container-id>

# Test with interactive mode
docker run -it --rm trend-model:local bash
```

### Health Check

The container includes a health check that verifies Streamlit is running:
```bash
curl http://localhost:8501/_stcore/health
```

Expected response: HTTP 200 OK

## Advanced Usage

### Custom Configuration

Run with your own configuration file:

```bash
# Mount config directory and specify custom config
docker run -p 8501:8501 -v $(pwd)/my-configs:/app/config \
  trend-model:local \
  streamlit run src/trend_portfolio_app/app.py
```

### Data Persistence

To persist results between container runs:

```bash
# Create a persistent volume for results
docker run -p 8501:8501 -v trend-data:/app/results \
  trend-model:local
```

### Container Customization

Override the default command for specific use cases:

```bash
# Run only the analysis without web interface
docker run --rm -v $(pwd):/workspace trend-model:local \
  python -m trend_analysis.run_analysis -c /workspace/my-config.yml

# Start Jupyter notebook server instead
docker run -p 8888:8888 trend-model:local \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

As with earlier sections, swap `trend-model:local` for the published
`ghcr.io/stranske/trend-model:latest` tag if you have not built locally.

## Support

- **Documentation**: See README.md for detailed usage instructions
- **Issues**: Report problems at https://github.com/stranske/Trend_Model_Project/issues
- **Source Code**: https://github.com/stranske/Trend_Model_Project

The Docker image is automatically built and updated from the latest code in the repository.