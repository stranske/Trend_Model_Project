# Trend Universe dataset

The Trend universe lives in two CSVs stored at the repository root:

- `Trend Universe Data.csv` – monthly total return series by fund/index.
- `Trend Universe Membership.csv` – the dated membership ledger for every column in the returns file.

## Membership convention

Each row in `Trend Universe Membership.csv` describes one contiguous membership window for a fund:

| column | required | description |
| --- | --- | --- |
| `fund` | ✅ | Exact column name from the returns CSV. |
| `effective_date` | ✅ | Inclusive start date for the fund. Returns that pre-date this value are ignored. |
| `end_date` | optional | Inclusive last date. Leave blank to keep the fund active through the latest sample. |

Funds can appear multiple times when they temporarily exit and later rejoin the universe. The loader sorts the windows per fund and nulls any returns that fall outside the active windows before the backtest runs.

## Extending the membership file

1. Copy the fund name exactly as it appears in `Trend Universe Data.csv`.
2. Add a new row with the desired `effective_date` (month-end) and optional `end_date`.
3. Keep rows sorted by `effective_date` for readability; the loader will enforce ordering when parsing.
4. Re-run the multi-period test suite to confirm the new membership dates are respected.

When a membership row omits `effective_date` the loader raises an error so gaps are surfaced early.
