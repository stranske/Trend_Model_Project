# Walk-forward (rolling OOS) analysis

The Results page in the Streamlit app includes a Walk-forward analysis expander that lets you:

- Set Train size, Test size, and Step (row counts)
- Optionally infer regimes from the sign of portfolio returns (+/-)
- Toggle views: Full period aggregate, OOS-only aggregate, Per‑regime (OOS)

This uses `trend_analysis.engine.walkforward.walk_forward` to aggregate a chosen metric across rolling out-of-sample windows and, when enabled, by regime.

CLI alternative:

```bash
python scripts/walkforward_cli.py --csv demo/demo_returns.csv --train 12 --test 3 --step 3 --column Portfolio
```

This prints full-period, OOS-only, and per-regime aggregates for quick experiments.
