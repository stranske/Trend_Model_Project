# Archived configuration templates

The files in this folder were retired after the current pipeline examples replaced their behavior.

## Mapping
- `rolling_hold_bayes.yml` → use the maintained walk-forward example in `config/walk_forward.yml`, which covers multi-period selection and grid search without the legacy threshold-hold policy.
- `cv_example.yml` → use the modern cross-validation + grid example in `config/walk_forward.yml` or the full backtest in `config/long_backtest.yml` depending on the workflow.
