# Walk-forward (rolling OOS) analysis

The Results page in the Streamlit app includes a **Walk-forward analysis** expander that now delivers a full drill-down of rolling windows:

- Configure **Train**, **Test**, and **Step** sizes (row counts).
- Supply regime labels by inferring the portfolio sign, or upload a CSV with custom tags (requires a `Date` column plus one or more label columns).
- Inspect three coordinated views:
  - **Full period** – aggregated statistics (mean + information ratio by default) for the entire backtest.
  - **OOS only** – out-of-sample summary plus a per-window table showing train/test boundaries and window-level statistics, with charts that pivot by statistic.
  - **Per regime** – OOS aggregates grouped by regime, including the information ratio per tag, displayed as both tables and charts.
- The engine infers the data frequency and displays the detected periods-per-year multiplier used for information-ratio annualisation.

Behind the scenes the page calls `trend_analysis.engine.walkforward.walk_forward`, which returns:

- Full-period and OOS summary tables (statistics × metrics) including information ratios.
- A per-window DataFrame capturing train/test date ranges, sample counts, and OOS statistics for each split.
- Optional per-regime aggregates, again with information ratios.

## CLI alternative

The helper script mirrors the UI functionality:

```bash
python scripts/walkforward_cli.py \
  --csv demo/demo_returns.csv \
  --train 12 --test 3 --step 3 \
  --column Portfolio \
  --regime-csv demo/regimes.csv --regime-column Regime
```

The CLI prints the inferred periods-per-year multiplier, full and OOS summaries, the per-window breakdown, and – when provided – per-regime aggregates.

## Numeric worked example

Suppose you run walk-forward with the following synthetic returns and regime tags:

| Date       | Portfolio | Regime |
|------------|-----------|--------|
| 2022-01-01 | 0.02      | Bull   |
| 2022-01-02 | 0.01      | Bull   |
| 2022-01-03 | -0.01     | Bear   |
| 2022-01-04 | 0.03      | Bull   |
| 2022-01-05 | -0.02     | Bear   |
| 2022-01-06 | 0.01      | Bull   |

With train=3, test=2, step=2, the splits are:

1. Train: 2022-01-01 to 2022-01-03, Test: 2022-01-04 to 2022-01-05
2. Train: 2022-01-03 to 2022-01-05, Test: 2022-01-06

OOS summary (mean):

| Window | Test Dates         | Mean Return | IR (annualised) |
|--------|--------------------|-------------|-----------------|
| 1      | 2022-01-04,05      | 0.005       | 0.35            |
| 2      | 2022-01-06         | 0.01        | 0.70            |

Per-regime OOS IR:

| Regime | OOS Dates          | Mean Return | IR (annualised) |
|--------|--------------------|-------------|-----------------|
| Bull   | 2022-01-04,06      | 0.02        | 0.80            |
| Bear   | 2022-01-05         | -0.02       | -0.40           |

These tables are produced by the Results page and CLI, and match the output of `walk_forward`.
