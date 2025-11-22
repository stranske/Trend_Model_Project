# Robust Portfolio Weighting - Implementation Guide

## Overview

This implementation adds robustness and structure upgrades to handle ill-conditioned covariances in portfolio construction. The system now includes shrinkage options (Ledoit-Wolf/OAS) and automatic fallback to "safe mode" methods when condition numbers are too high.

## New Features

### 1. Robust Weight Engines

#### RobustMeanVariance (`robust_mv`, `robust_mean_variance`)
Mean-variance optimization with shrinkage and safe mode fallback:

```python
from trend_analysis.plugins import create_weight_engine

engine = create_weight_engine(
    "robust_mv",
    shrinkage_method="ledoit_wolf",      # none | ledoit_wolf | oas
    condition_threshold=1e12,             # max allowed condition number
    safe_mode="hrp",                     # hrp | risk_parity | diagonal_mv
    diagonal_loading_factor=1e-6,        # regularization factor
    min_weight=0.0,                      # weight constraints
    max_weight=1.0
)
```

#### RobustRiskParity (`robust_risk_parity`)
Enhanced risk parity with condition monitoring:

```python
engine = create_weight_engine(
    "robust_risk_parity",
    condition_threshold=1e12,
    diagonal_loading_factor=1e-6
)
```

### 2. Enhanced Existing Engines

All existing engines (RiskParity, HRP, ERC) now include:
- Robustness checks for numerical stability
- Comprehensive logging of decisions
- Graceful degradation for pathological inputs

### 3. Configuration Schema Extensions

Add robustness settings to your configuration:

```yaml
portfolio:
  weighting_scheme: "robust_mv"  # or robust_risk_parity
  
  # New robustness section
  robustness:
    # Covariance shrinkage options
    shrinkage:
      enabled: true
      method: "ledoit_wolf"          # none | ledoit_wolf | oas
    
    # Condition number monitoring and safe mode fallback
    condition_check:
      enabled: true
      threshold: 1.0e12              # maximum allowed condition number
      safe_mode: "hrp"               # hrp | risk_parity | diagonal_mv
      diagonal_loading_factor: 1.0e-6
    
    # Logging configuration for robustness decisions
    logging:
      log_method_switches: true      # log when switching to safe mode
      log_shrinkage_intensity: true  # log shrinkage parameters
      log_condition_numbers: true    # log matrix condition numbers
```

## Usage Examples

### Basic Usage

```python
import pandas as pd
from trend_analysis.plugins import create_weight_engine

# Create covariance matrix
cov = pd.DataFrame([[0.04, 0.002], [0.002, 0.09]], 
                   index=['A', 'B'], columns=['A', 'B'])

# Use robust engine
engine = create_weight_engine("robust_mv")
weights = engine.weight(cov)
print(weights)
```

### Handling Pathological Cases

The system automatically handles:

1. **Singular matrices** (perfectly correlated assets)
2. **Ill-conditioned matrices** (near-singular)
3. **Non-positive definite matrices**
4. **Extreme variance differences**
5. **Zero variances**

### Logging Output

With appropriate logging configuration, you'll see:

```
DEBUG: Applied Ledoit-Wolf shrinkage with intensity 0.2341
DEBUG: Covariance matrix condition number: 1.23e+08
WARNING: Ill-conditioned covariance matrix (condition number: 1.23e+15 > threshold: 1.00e+12). 
         Switching to safe mode: hrp
INFO: Using mean-variance optimization with ledoit_wolf shrinkage
```

## Configuration Files

### Demo Configuration
Use `config/robust_demo.yml` for testing robustness features:

```bash
PYTHONPATH="./src" python -m trend_analysis.run_analysis -c config/robust_demo.yml
```

### Production Configuration
Update `config/defaults.yml` with your preferred robustness settings.

## Testing

Comprehensive test suite in `tests/test_robust_weighting.py`:

```bash
# Run robustness tests (requires dependencies)
pytest tests/test_robust_weighting.py -v

# Test syntax (no dependencies needed)
python3 -m py_compile src/trend_analysis/weights/robust_weighting.py
```

## Migration Guide

### Existing Configurations
All existing configurations remain compatible. The robustness features are opt-in.

### Upgrading to Robust Engines
1. Change `weighting_scheme` to `robust_mv` or `robust_risk_parity`
2. Add `robustness` section to configuration
3. Adjust logging level to see robustness decisions

### Gradual Migration
1. Start with existing engines (now enhanced with robustness checks)
2. Test with `robust_demo.yml` configuration
3. Gradually migrate to robust engines in production

## Performance Considerations

- **Shrinkage**: Minimal computational overhead
- **Condition monitoring**: O(n³) for eigenvalue computation
- **Safe mode fallback**: HRP is more expensive than risk parity
- **Logging**: Negligible impact when set to appropriate level

## Troubleshooting

### Common Issues
1. **ImportError**: Ensure all dependencies (numpy, pandas, scipy) are installed
2. **Configuration errors**: Validate YAML syntax with `python -c "import yaml; yaml.safe_load(open('config.yml'))"`
3. **Convergence issues**: Check ERC max_iter and tolerance settings

### Debug Mode
Enable detailed logging:

```yaml
run:
  log_level: "DEBUG"
```

## Advanced Usage

### Custom Shrinkage
Implement custom shrinkage methods by extending the shrinkage functions in `robust_weighting.py`.

### Custom Safe Modes
Add new safe mode methods by extending the `_safe_mode_weights` method.

### Integration with Existing Pipeline
The robust engines integrate seamlessly with the existing portfolio construction pipeline through the plugin system.

## References

- **Ledoit & Wolf (2004)**: "A well-conditioned estimator for large-dimensional covariance matrices"
- **Chen et al. (2010)**: "Shrinkage algorithms for MMSE covariance estimation"
- **Lopez de Prado (2016)**: "Building Diversified Portfolios that Outperform Out-of-Sample" (HRP)

## Support

For issues or questions:
1. Check test cases in `tests/test_robust_weighting.py` for examples
2. Run `python examples/demo_robust_weighting.py` for a full demonstration
3. Review logging output for robustness decisions