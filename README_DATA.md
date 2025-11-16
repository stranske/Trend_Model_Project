# Demo dataset provenance

This repository bundles a few small CSV files (for example `hedge_fund_returns_with_indexes.csv`, `Trend Universe Data.csv`, and other samples under `demo/`). They exist solely to support tests, docs, and interactive demos.

- **Provenance:** All bundled data is synthetic or derived from public benchmark series. No confidential client information or proprietary hedge fund records are present.
- **Intended use:** The files allow contributors to exercise the Trend Model demos, verify export pipelines, and run automated tests without reaching out to live data providers.
- **Limitations & restrictions:**
  - The distributions and correlations are intentionally simplified. Do **not** rely on them for research, production trading, or risk disclosures.
  - Because the numbers are synthetic/public, they do not represent actual manager performance and must not be marketed as such.
  - Keep the files inside this repository; if you redistribute them, retain this notice so downstream users understand the limitations.

If you need richer datasets, regenerate them locally via the scripts in `scripts/` or plug in your own data sources.

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
