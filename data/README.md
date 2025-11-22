# Reference datasets

Only active reference inputs remain in the root or `data/` directories. Demo outputs and ad-hoc analyses are archived under `archives/data_snapshots/`.

| Dataset | Location | Size | Schema | Owner | Purpose |
| --- | --- | --- | --- | --- | --- |
| Trend Universe total returns | `Trend Universe Data.csv` | 136 KB | [Schema](#trend-universe-data) | Demo data maintainers (Research Ops) | Primary monthly total return matrix for Trend universe configs and tests. |
| Trend Universe membership ledger | `Trend Universe Membership.csv` | 4 KB | [Schema](#trend-universe-membership) | Demo data maintainers (Research Ops) | Effective-date windows for each Trend universe column used by loaders and configs. |
| Hedge fund returns with benchmarks | `hedge_fund_returns_with_indexes.csv` | 140 KB | [Schema](#hedge-fund-returns-with-indexes) | Demo data maintainers (Research Ops) | Input for long backtest and rolling-hold configs plus legacy demo notebooks. |
| Sample manager returns | `data/raw/managers/sample_manager.csv` | 4 KB | [Schema](#sample-manager-returns) | Demo data maintainers (Research Ops) | Minimal fixture used by CLI preset tests. |
| Sample benchmark index | `data/raw/indices/sample_index.csv` | 4 KB | [Schema](#sample-benchmark-index) | Demo data maintainers (Research Ops) | Lightweight index series kept for parity with sample manager inputs. |

## Schemas

### Trend Universe Data
- Columns: `Date` monthly period followed by fund/index total return series (percentage returns as decimals) for the Trend universe.
- Frequency: Monthly rows.
- Pair with the membership ledger to filter each fund to its effective window.

### Trend Universe Membership
- Columns: `fund`, `effective_date`, `end_date`.
- Each row encodes a contiguous active window for the corresponding return column in the Trend universe file. Missing `end_date` means the fund remains active through the latest sample.

### Hedge fund returns with indexes
- Columns: `Date`, `Risk-Free Rate`, and multiple fund/index total return series (percentage returns as decimals).
- Frequency: Monthly rows designed for demo backtests and rolling-hold examples.

### Sample manager returns
- Columns: `Date`, `Fund_A`, `Fund_B` monthly total return series.
- Frequency: Monthly rows used for CLI trend preset tests.

### Sample benchmark index
- Columns: `Date`, `SPX` monthly index level or return series (demo-scale values).
- Frequency: Monthly rows aligned with the sample manager dataset.
