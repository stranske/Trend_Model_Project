# Active shipped configuration alias inventory

Inventory for verifier follow-up #5875. Lists aliases still present in shipped
`config/**` payloads that select alternate runtime branches. Legacy
`trend_analysis.config.legacy` is retired; the canonical loader owns validation.

| Alias / key | Shipped in | Runtime consumer | Status |
|-------------|------------|------------------|--------|
| `portfolio.weighting.name` | `config/defaults.yml` | Canonical weighting resolver | Active; `portfolio.weighting_scheme` is the UI path |
| `portfolio.weighting_scheme` | `config/defaults.yml`, `config/demo.yml` | Portfolio construction | Active |
| `portfolio.selector.name` / `params` | `config/demo.yml` | Rank selection | Active in demo payloads only |
| `portfolio.custom_weights` | `config/demo.yml` | Demo weight override | Active in demo payloads only |
| `data.missing_policy` / `missing_limit` | `config/defaults.yml` | Data ingestion | Active canonical surface |
| `preprocessing.missing_data.*` | `config/defaults.yml`, `config/demo.yml` | Preprocessing stage | Active; distinct from top-level `data.missing_*` |
| `vol_adjust.enabled` | `config/defaults.yml`, `config/demo.yml` | Vol targeting gate | Active; UI maps via `vol_adjust_enabled` |
| `run.jobs` | `config/defaults.yml`, `config/demo.yml` | Parallel execution | Active |
| `run.monthly_cost` | `config/defaults.yml` | Single-period cost path | Active shipped default (0.0) |
| `multi_period.frequency` (`ME` alias) | `config/demo.yml` | Multi-period scheduler | Active month-end alias |
| `sample_split` (`in_start`/`out_start` keys) | `config/demo.yml` | Demo split | Active demo-only shape |

No shipped payload under `config/` references `config.legacy`, `nan_policy`,
top-level `jobs`, or removed weighting keys. The regression gate is:

```bash
rg 'config\.legacy|from \.legacy' src streamlit_app scripts tests docs \
  --glob '!docs/archive/**' --glob '!docs/keepalive/**'
```
