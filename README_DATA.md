# Demo dataset provenance

This repository bundles a few small CSV files (for example `hedge_fund_returns_with_indexes.csv`, `Trend Universe Data.csv`, and other samples under `demo/`). They exist solely to support tests, docs, and interactive demos.

- **Provenance:** All bundled data is synthetic or derived from public benchmark series. No confidential client information or proprietary hedge fund records are present.
- **Intended use:** The files allow contributors to exercise the Trend Model demos, verify export pipelines, and run automated tests without reaching out to live data providers.
- **Limitations & restrictions:**
  - The distributions and correlations are intentionally simplified. Do **not** rely on them for research, production trading, or risk disclosures.
  - Because the numbers are synthetic/public, they do not represent actual manager performance and must not be marketed as such.
  - Keep the files inside this repository; if you redistribute them, retain this notice so downstream users understand the limitations.

If you need richer datasets, regenerate them locally via the scripts in `scripts/` or plug in your own data sources.
