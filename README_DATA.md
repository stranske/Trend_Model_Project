# Demo data provenance

This repository includes a handful of CSV files used by the examples and
regression tests, such as:

- `Trend Universe Data.csv`
- `Trend Universe Membership.csv`
- `hedge_fund_returns_with_indexes.csv`
- files under `data/raw/indices/` and `data/raw/managers/`

## Where the data comes from

All of the committed CSVs are synthetic or aggregated from publicly available
benchmark information.  They are generated specifically for Trend Model demos
and CI smoke tests so contributors have predictable inputs.  No proprietary or
client-contributed datasets are stored in this repository.

## Intended use

The files are safe to use for:

- running the quickstart and stress-test demos
- validating CI workflows and regression suites
- experimenting locally with the example scripts

They are **not** a drop-in replacement for production data sources, and they are
not meant to inform live investment decisions.

## Limitations and restrictions

- Shapes, magnitudes, and relationships are tuned for documentation purposes,
  so performance metrics from these CSVs should not be interpreted as real-world
  results.
- When building new features, avoid encoding assumptions that only hold for
  these synthetic samples (for example, fixed column ordering or the limited set
  of tickers).
- Do not redistribute the files as if they were representative research data;
  they are strictly illustrative.

If you need to extend the dataset for a new tutorial, keep the data synthetic or
pull it from clearly documented public sources, then update this README to note
any additions.
