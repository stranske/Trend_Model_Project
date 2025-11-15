# Docker Quickstart Guide

This guide provides zero-setup local runs using the pre-built Docker container.

## Quick Start

### Option 1: Run the Web Interface (Recommended)

Launch the Streamlit web application with a single command:

```bash
docker run -p 8501:8501 ghcr.io/stranske/trend-model:latest
```

Then open http://localhost:8501 in your browser to access the interactive trend analysis application.

### Option 2: Use the CLI

Run analysis directly from command line:

```bash
# Generate demo data and run analysis
docker run --rm ghcr.io/stranske/trend-model:latest python scripts/generate_demo.py

# Run trend analysis with demo config
docker run --rm -v $(pwd):/workspace ghcr.io/stranske/trend-model:latest \
  trend-analysis -c config/demo.yml
```

### Option 3: Interactive Shell

Get a shell inside the container for development:

```bash
docker run -it --rm ghcr.io/stranske/trend-model:latest bash
```

### Option 4: Docker Compose (App + API)

Bring up the Streamlit app and a minimal API server with one command:

```bash
docker compose up app
```

This uses the included `docker-compose.yml` which also mounts `./data` into the
container at `/app/data` for local files.

## Working with Your Data

To analyze your own data, mount a local directory containing your CSV files:

```bash
docker run -p 8501:8501 -v /path/to/your/data:/app/data ghcr.io/stranske/trend-model:latest
```

Your data will be accessible at `/app/data/` inside the container.

## Container Features

### Included Components
- **Streamlit Web App**: Full-featured interactive interface at port 8501
- **CLI Tools**: `trend-analysis`, `trend-multi-analysis`, and `trend-model` commands
- **Demo Data**: Built-in demo datasets for testing
- **All Dependencies**: Pandas, NumPy, PyYAML, and all required packages

### Security Features
- **Non-root User**: Runs as user `appuser` (UID 1001) for security
- **Minimal Base**: Python 3.11 slim image with only essential system packages
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

1. **Build locally** (if modifying the container):
   ```bash
   git clone https://github.com/stranske/Trend_Model_Project.git
   cd Trend_Model_Project
   make lock  # ensure requirements.lock reflects pyproject.toml
   docker build -t trend-model-dev .
   ```

2. **Run with live code updates**:
   ```bash
   docker run -p 8501:8501 -v $(pwd):/app trend-model-dev
   ```

### Testing the Container

Verify the container works correctly:

```bash
# Test web interface starts correctly
docker run -d --name trend-test -p 8501:8501 ghcr.io/stranske/trend-model:latest
sleep 10
curl -f http://localhost:8501/_stcore/health
docker stop trend-test && docker rm trend-test

# Test CLI functionality
docker run --rm ghcr.io/stranske/trend-model:latest trend-analysis --help

# Comprehensive test suite
./test_docker.sh
```

The Dockerfile installs dependencies via `uv pip sync --system requirements.lock`
followed by `pip install --no-deps -e .[app]`, so keeping the lock file current
is essential for reproducible builds.

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Use a different port
docker run -p 8502:8501 ghcr.io/stranske/trend-model:latest
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
docker run -it --rm ghcr.io/stranske/trend-model:latest bash
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
  ghcr.io/stranske/trend-model:latest \
  streamlit run src/trend_portfolio_app/app.py
```

### Data Persistence

To persist results between container runs:

```bash
# Create a persistent volume for results
docker run -p 8501:8501 -v trend-data:/app/results \
  ghcr.io/stranske/trend-model:latest
```

### Container Customization

Override the default command for specific use cases:

```bash
# Run only the analysis without web interface
docker run --rm -v $(pwd):/workspace ghcr.io/stranske/trend-model:latest \
  python -m trend_analysis.run_analysis -c /workspace/my-config.yml

# Start Jupyter notebook server instead
docker run -p 8888:8888 ghcr.io/stranske/trend-model:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

## Support

- **Documentation**: See README.md for detailed usage instructions
- **Issues**: Report problems at https://github.com/stranske/Trend_Model_Project/issues
- **Source Code**: https://github.com/stranske/Trend_Model_Project

The Docker image is automatically built and updated from the latest code in the repository.