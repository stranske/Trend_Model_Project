# Monte Carlo Visualization Contract

This document defines the `nav_paths` data contract used by the visualization layer and the expected extension workflow for new charts.

## `nav_paths` Schema

### Accepted input to `make_paths(nav_paths)`

`nav_paths` must be a `pandas.DataFrame` with:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| index | datetime-like index | yes | Converted with `pd.to_datetime`; non-datetime values raise `ValueError`. |
| columns | path labels or `MultiIndex` | yes | Plain path labels are accepted. For `MultiIndex`, a `path` level is used; if an `asset` level exists, only `asset == "NAV"` is retained. |
| values | numeric-like | yes | Coerced via `pd.to_numeric(errors="coerce")`. |

### Canonical output from `make_paths(nav_paths)`

The adapter returns a long-form frame with:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| index level `date` | `datetime64[ns]` | yes | First level of canonical `MultiIndex`. |
| index level `path` | string/int label | yes | Second level of canonical `MultiIndex`. |
| `nav` | `float64` | yes | Required value column used by downstream charts. |

Contract constants are defined in `src/trend_analysis/viz/adapters.py`:
- `PATHS_INDEX_NAMES = ("date", "path")`
- `PATHS_REQUIRED_COLUMNS = ("nav",)`
- `PATHS_REQUIRED_DTYPES = {"nav": "float64"}`

## Chart Extension Points

### Adapter layer

Use the adapter layer to keep chart modules stable against upstream shape changes:

- `make_paths(nav_paths)` normalizes raw navigation paths.
- `terminal_returns(paths)` calculates per-path terminal returns.
- `rolling_stats(paths)` emits rolling mean/std/Sharpe diagnostics.
- `path_correlations(paths)` emits a correlation matrix.

### Chart modules

Core chart constructors currently used for MC diagnostics:

- `trend_analysis.viz.fan.make`
- `trend_analysis.viz.path_dist.make`
- `trend_analysis.viz.risk_return.make`

When adding a chart:

1. Build from adapter outputs rather than raw bundle objects.
2. Return a Plotly `go.Figure`.
3. Add smoke coverage that figure creation succeeds and `fig.to_json()` is valid JSON.
4. Add focused unit tests for any schema assumptions in the chart.

## Triggering Agent Work After Chart Creation

Use this flow after introducing or changing charts:

1. Run focused tests.
   - `pytest tests/test_visualization_smoke.py tests/test_nav_paths_contract.py tests/test_nav_paths_adapters.py -m "not slow"`
2. For export behavior without Kaleido, run:
   - `pytest tests/test_export_without_kaleido.py -k kaleido -m "not slow"`
3. Post a keepalive/agent follow-up request in the PR with explicit next tasks and acceptance criteria.
   - Example task: `- [ ] Add test(s) for <new chart module> input validation.`
   - Example acceptance: `- [ ] pytest tests/test_<new_chart>.py exits with code 0.`

This keeps future agent rounds aligned to the same schema contract and verification commands.
