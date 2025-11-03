<!-- bootstrap for codex on issue #3233 -->

## Scope

- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Task list

- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [ ] `__init__.py`
  - [ ] `data.py`
  - [ ] `presets.py`
  - [ ] `harness.py`
  - [ ] `regimes.py`
  - [ ] `pipeline.py`
  - [ ] `validators.py`
  - [ ] `run_analysis.py`
  - [ ] `market_data.py`
  - [ ] `signal_presets.py`
  - [ ] `frequency.py`
  - [ ] `signals.py`
  - [ ] `bootstrap.p`
  - [ ] `risk.py`
  - [ ] `bundle.py`
  - [ ] `cli.py`
  - [ ] `optimizer.py`
  - [ ] `model.py`
  - [ ] `engine.py`

## Acceptance criteria

- [ ] Test coverage exceeds 95% for each file.
- [ ] Essential functions for the program have full test coverage.

## Coverage snapshot

Latest manual “soft gate” coverage sampling (see pytest invocations recorded in workflow notes) surfaced the following modules below the 95% target. Modules already above the threshold are omitted for brevity.

| Module | Coverage | Notes |
| --- | --- | --- |
| `trend_analysis/backtesting/bootstrap.py` | 11% | `pytest tests/test_block_bootstrap_engine.py --cov=trend_analysis.backtesting.bootstrap --cov-report=term-missing`【b752fd†L1-L13】 |
| `trend_analysis/io/validators.py` | 77% | `pytest tests/test_validators.py tests/test_validators_extended.py --cov=trend_analysis.io.validators --cov-report=term-missing`【68497f†L1-L27】 |
| `trend_analysis/pipeline.py` | 87% | Combined pipeline-focused suite (`pytest tests -k "pipeline" --cov=trend_analysis.pipeline --cov-report=term-missing`) hit an existing live-docs autofix failure and left 63 statements uncovered.【8a1e92†L1-L159】 |
| `trend_analysis/io/market_data.py` | 89% | `pytest tests/test_market_data_validation.py tests/test_data.py tests/test_data_malformed_dates.py tests/test_data_schema.py --cov=trend_analysis.io.market_data --cov-report=term-missing`【05421d†L1-L28】 |
| `trend_analysis/cli.py` | 90% | `pytest tests/test_cli.py … tests/test_cli_trend_presets.py --cov=trend_analysis.cli --cov-report=term-missing` (multiple CLI suites).【1c10ea†L1-L27】 |
| `trend_analysis/multi_period/engine.py` | 91% | `pytest` against the multi-period engine suites with coverage.【1e7463†L1-L14】 |
| `trend_analysis/export/bundle.py` | 93% | `pytest tests/test_export_bundle.py --cov=trend_analysis.export.bundle --cov-report=term-missing`【da5792†L1-L16】 |
| `trend_analysis/config/model.py` | 93% | `pytest` across config model suites with coverage.【0a9354†L1-L16】 |
| `trend_analysis/engine/optimizer.py` | 95% | `pytest` across optimizer suites with coverage (just meets target but flagged for potential follow-up hardening).【02f673†L1-L13】 |

Representative modules already clearing the bar include:

- `trend_analysis/__init__.py` – 100%【7e5b05†L1-L64】
- `trend_analysis/data.py` – 96%【0afe9d†L1-L24】
- `trend_analysis/backtesting/harness.py` – 100%【db3091†L1-L22】
- `trend_analysis/regimes.py` – 98%【1cb9a4†L1-L24】
- `trend_analysis/run_analysis.py` – 96%【051720†L1-L24】
- `trend_analysis/signal_presets.py` – 100%【a92b1a†L1-L13】
- `trend_analysis/util/frequency.py` – 100%【ec84bc†L1-L13】
- `trend_analysis/signals.py` – 100%【4ee466†L1-L16】
- `trend_analysis/risk.py` – 97%【4ea963†L1-L13】

### Outstanding coverage blockers

- The pipeline-focused runs repeatedly fail on `test_autofix_pipeline_repairs_live_documents` because optional tooling (ruff, black, mypy, docformatter) rewrites expectations; manual reconciliation or sandboxing is needed before the suite can provide stable coverage.【8a1e92†L24-L159】
- High-touch modules (e.g. `market_data.py`, `validators.py`, `config/model.py`) still need targeted tests to exercise error paths, legacy option coercion, and IO plumbing so that coverage can rise above 95% without muting functionality.
