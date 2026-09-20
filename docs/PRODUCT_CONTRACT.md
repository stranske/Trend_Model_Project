# Product contract — stranske/Trend_Model_Project
_First draft generated 2026-09-20 from the audit scorecard; the repo owns this file from now on. A PR that adds a user-facing route, command or page adds a line here. The audit's Phase 1.5 scores every line below and prints any surface not listed as UNSCORED._

## Purpose
Turn fund-return data and configuration into a ranked, weighted trend portfolio/backtest and exportable analyst report.

## Primary journey
Provide returns and configuration → run model → inspect holdings and metrics → export analyst report → verify configuration/data changes affect results.

## Core functions
| id | a <user> can … and sees … | entry point | probe (how to exercise it; vary these determinants) | status 2026-09-20 |
|---|---|---|---|---|
| C1 | an analyst can run the trend model and sees ranked holdings, weights and portfolio metrics | `trend run -c CONFIG --returns CSV`; Streamlit | run demo returns with complete config; truncate to 72 rows and diff holdings/warnings | WORKS |
| C2 | an analyst can produce deliverables and sees HTML plus CSV, JSON, XLSX and TXT describing the run | `trend report -c CONFIG -i CSV --out DIR --output FILE`; GUI export | export baseline and enabled-volatility reports; compare parameters and artifacts | WORKS |
| C3 | an analyst can change configuration or data and sees results or validation change consistently | change lookback, volatility adjustment or CSV then rerun | vary `regime.lookback`, `vol_adjust`, and truncate returns; diff ranks, weights and metrics | FABRICATED |

## Known gaps at draft time
- C3: changing documented `regime.lookback` produces byte-identical holdings, weights and metrics without an explained invariant; that control is inert.
