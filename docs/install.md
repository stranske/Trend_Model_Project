# Installation Guide

This guide covers how to install and set up the Trend Model Project.

## Quick Start

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/stranske/Trend_Model_Project.git
cd Trend_Model_Project

# Set up the environment
./scripts/setup_env.sh

# Generate demo data
python scripts/generate_demo.py

# Run the analysis
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/demo.yml
```

## Prerequisites

- **Python 3.11+** (required)
- **Git** (for cloning)
- **Node.js v20+** (optional, for workflow tests)
- **uv** (optional, for fast dependency management)

## Installation Methods

### 1. Development Installation (Recommended)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install uv
uv pip sync requirements.lock

# Install package in editable mode
pip install --no-deps -e '.[dev]'
```

### 2. Using setup script

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

### 3. Docker (No local setup needed)

```bash
docker run -p 8501:8501 ghcr.io/stranske/trend-model:latest
```

Visit http://localhost:8501 after the container starts.

## Verification

After installation, verify everything works:

```bash
# Run tests
./scripts/run_tests.sh

# Check CLI
PYTHONPATH="./src" python -m trend_analysis.run_analysis --help

# Launch Streamlit app
./scripts/run_streamlit.sh
```

## Next Steps

- See [quickstart.md](quickstart.md) for a 10-minute tutorial
- See [UserGuide.md](UserGuide.md) for comprehensive documentation
- See [CLI.md](CLI.md) for command-line interface reference
