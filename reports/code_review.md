# Code Review Findings (targeted)

## Potential failures / fragility
- `pipeline._build_sample_windows` still relies on `pd.api.types.is_datetime64tz_dtype`, which pandas has flagged for removal. Once the API disappears, timezone-aware datasets will raise at window construction before any diagnostics are produced. Consider switching to `isinstance(dtype, pd.DatetimeTZDtype)` (or `DatetimeTZDtype.is_dtype`) to future-proof the check while keeping behaviour identical.【F:src/trend_analysis/pipeline.py†L379-L404】

- Market data cadence validation wraps calculations in the deprecated `mode.use_inf_as_na` option. Pandas plans to drop the flag, so this block will start raising before the frequency checks finish. Normalising `inf` values ahead of time (e.g., `Series.replace([np.inf, -np.inf], np.nan)`) would avoid the dependency and keep the validation path stable.【F:src/trend_analysis/io/market_data.py†L429-L449】

## Design / correctness concerns
- `analysis.results.compute_universe_fingerprint` silently treats missing or unreadable data/membership files as empty byte streams. That produces the same fingerprint for "no files" and "empty files", so runs can look reproducible even when input data are absent or corrupt. Surfacing a warning or distinguishing the missing-data case in the hash payload would make the fingerprint more trustworthy.【F:analysis/results.py†L31-L66】

## Improvement opportunities
- The multi-period/unit tests surfaced deprecation warnings around monthly frequency handling (e.g., `pd.date_range(..., freq="M")`). Similar month-end assumptions appear in `_build_sample_windows` via `pd.Period(text, freq="M")`. Updating to the new pandas month-end aliases (`"ME"`) would quiet the warnings and reduce future upgrade risk while keeping the calendar logic intact.【F:src/trend_analysis/pipeline.py†L375-L404】【499981†L1-L21】

- Weight-engine failures in `_compute_weights_and_stats` are swallowed and converted into equal-weight fallbacks with only a warning. That keeps runs alive but also hides real configuration or numerical issues from callers. Returning the fallback info through the diagnostic payload (or propagating a structured error) would make downstream consumers aware that the requested weighting scheme was bypassed.【F:src/trend_analysis/pipeline.py†L713-L747】
