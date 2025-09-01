# Reproducibility Guide

This guide explains how to ensure reproducible results when running the Trend Model Project, including handling of random seeding and hash randomization.

## Random Seeding

The Trend Model Project uses random numbers in several contexts:
- Portfolio selection (`selection_mode: "random"`)
- Monte Carlo simulations
- Bayesian weighting algorithms

To ensure reproducible results, you can set the `random_seed` parameter in your configuration:

```yaml
portfolio:
  random_seed: 42
```

## Hash Randomization and PYTHONHASHSEED

Python uses hash randomization by default for security reasons, which can affect the ordering of dictionaries and sets. This can impact reproducibility in some scenarios.

### Important Limitation

**Setting `PYTHONHASHSEED` after the Python interpreter has started has no effect.** This environment variable must be set **before** Python starts to control hash randomization.

Some dependencies (including NumPy's test suite) attempt to set `PYTHONHASHSEED` during test execution:

```python
# This is INEFFECTIVE - Python is already running
monkeypatch.setenv('PYTHONHASHSEED', '0')
```

### Proper Hash Seeding Setup

To ensure reproducible hash behavior across runs:

1. **Set `PYTHONHASHSEED` before starting Python:**
   ```bash
   export PYTHONHASHSEED=0
   python -m trend_analysis.run_analysis -c config/demo.yml
   ```

2. **Use in test environments:**
   ```bash
   export PYTHONHASHSEED=0
   ./scripts/run_tests.sh
   ```

3. **Add to shell scripts that need reproducibility:**
   ```bash
   #!/bin/bash
   export PYTHONHASHSEED=0
   exec python -m trend_analysis.run_analysis "$@"
   ```

### CI/CD Considerations

For continuous integration environments, ensure hash seeding is set at the environment level:

```yaml
# Example GitHub Actions
env:
  PYTHONHASHSEED: 0
```

```yaml
# Example environment configuration
environment:
  variables:
    PYTHONHASHSEED: "0"
```

## Best Practices for Reproducible Results

1. **Always set random_seed in configuration files**
2. **Set PYTHONHASHSEED=0 before starting Python** (not during runtime)
3. **Document seed values used in published results**
4. **Use version pinning for critical dependencies**
5. **Test reproducibility across different environments**

## Testing Reproducibility

To verify that your setup produces reproducible results:

```bash
# Run 1
export PYTHONHASHSEED=0
python -m trend_analysis.run_analysis -c config/demo.yml > results1.txt

# Run 2  
export PYTHONHASHSEED=0
python -m trend_analysis.run_analysis -c config/demo.yml > results2.txt

# Compare
diff results1.txt results2.txt
```

Results should be identical when using the same configuration and environment setup.

## Troubleshooting

- **Results vary between runs**: Ensure `PYTHONHASHSEED` is set before Python starts
- **Tests show warnings about hash seeding**: This is expected from NumPy's test suite and can be safely ignored
- **Different results on different systems**: Check Python version, library versions, and floating-point precision