# Active shipped configuration alias inventory

Inventory for verifier follow-up #5875. Lists aliases still present in shipped
`config/**` payloads that select alternate runtime branches. Legacy
`trend_analysis.config.legacy` is retired; the canonical loader owns validation.

| Alias / key | Shipped in | Runtime consumer | Status |
|-------------|------------|------------------|--------|
| `portfolio.weighting.name` | `defaults.yml`, `demo.yml`, `long_backtest.yml`, `portfolio_test.yml`, `robust_demo.yml`, `trend_concentrated_2004.yml`, `trend_universe_2004.yml` | Canonical weighting resolver | Active; `portfolio.weighting_scheme` is the UI path |
| `portfolio.weighting_scheme` | `defaults.yml`, `demo.yml`, `robust_demo.yml`, and Monte Carlo strategy payloads | Portfolio construction | Active |
| `portfolio.selector.name` / `params` | `demo.yml`, `long_backtest.yml`, `portfolio_test.yml`, `robust_demo.yml`, `trend_concentrated_2004.yml`, `trend_universe_2004.yml` | Rank selection | Active shipped selector shape |
| `portfolio.custom_weights` | `demo.yml` | Demo weight override | Active demo-only shape |
| `data.missing_policy` / `missing_limit` | `defaults.yml`, `demo.yml`, `robust_demo.yml` | Data ingestion | Active canonical surface |
| `preprocessing.missing_data.*` | `defaults.yml`, `demo.yml` | Preprocessing stage | Active; distinct from top-level `data.missing_*` |
| `vol_adjust.enabled` | `defaults.yml`, `demo.yml`, `robust_demo.yml` | Vol targeting gate | Active; UI maps via `vol_adjust_enabled` |
| `run.jobs` | Core presets plus scenario payloads | Parallel execution | Active; the deprecated top-level `jobs` key is forbidden |
| `run.monthly_cost` | `defaults.yml` | Single-period cost path | Active shipped default (0.0) |
| `multi_period.frequency` (`ME` alias) | `demo.yml`, `long_backtest.yml`, `trend_concentrated_2004.yml`, `trend_universe_2004.yml` | Multi-period scheduler | Active month-end alias |
| `sample_split` (`in_start`/`out_start` keys) | `config/demo.yml` | Demo split | Active demo-only shape |

No shipped YAML payload under `config/` uses the retired legacy-import marker,
the `nan_policy` alias, top-level `jobs`, or retired flat weighting keys. The
following gate intentionally scans only shipped configuration: source and
documentation contain compatibility and explanatory references that are not
payload keys.

```bash
rg_exit=0
rg -n -e 'config\.legacy|from \.legacy|(^|[[:space:]])nan_policy:|^jobs:|weighting_(name|method):' config --glob '*.yml' --glob '*.yaml' || rg_exit=$?
case "$rg_exit" in
  0) echo "forbidden shipped configuration alias found" >&2; exit 1 ;;
  1) ;; # rg reports no matches with status 1; that is the passing result.
  *) exit "$rg_exit" ;;
esac
```
