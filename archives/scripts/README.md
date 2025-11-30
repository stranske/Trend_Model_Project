# Archived root-level scripts

The following root-level scripts were moved here after a repository sweep.
`rg` found no references to them in the `Makefile`, `docs/`, or `scripts/`
directories, so they are treated as rarely used or historical utilities.

- `demo_malformed_date_fix.py` — manual demo of malformed-date validation
  behavior. Last-known use: validation fix walkthrough; not referenced by
  automation.
- `demo_proxy.py` — Streamlit WebSocket proxy demonstration. Last-known use:
  manual debugging of the Streamlit proxy; no active automation references.
- `manager_attribution_analysis.py` — deprecated wrapper superseded by the
  unified `trend` CLI. Last-known use: pre-unified CLI era; retained only as
  a reference.

## 2025-11-30-one-off

- `demo_export_fix.py` — one-off demo export fix script; no references found
  in workflows, docs, or other scripts.

If you need to revive any of these scripts, please re-home them under a
supported workflow and add the appropriate documentation and ownership notes.
