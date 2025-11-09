# Soft coverage findings

Results from `coverage run -m pytest` followed by `coverage report -m` on 2025-02-14.

| File | Coverage | Missed stmts | Missed branches | Notes |
|------|----------|--------------|-----------------|-------|
| `src/trend/reporting/unified.py` | 88% | 51 | 50 | Large blocks of PDF export helpers remain untested. |
| `src/trend_analysis/multi_period/engine.py` | 92% | 29 | 24 | Gaps cluster around schedule rollback and resume branches. |
| `src/trend/cli.py` | 94% | 7 | 17 | Missing coverage concentrated in fallback command paths. |
| `src/trend_portfolio_app/app.py` | 94% | 12 | 10 | Streamlit run orchestration still lacks error-path assertions. |

All other measured modules meet or exceed the 95% target in this sweep. Pending work should focus on the four files above.
