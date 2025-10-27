# ChatGPT Issue Sync via Agents 63 Issue Intake

_Last updated: 2026-02-21_

This document explains how to run the `agents-63-issue-intake.yml` GitHub Actions workflow in **ChatGPT sync mode**. The workflow ingests topic lists and synchronizes them to GitHub issues with strong normalization, diagnostics, and fallback behavior. Manual dispatch now happens directly from the intake workflow—set `intake_mode` to `chatgpt_sync` and supply one of the supported topic sources.

## Purpose
Convert a (potentially long) enumerated list of topic ideas generated in ChatGPT into individual, consistently labeled GitHub issues. The workflow hardens ingestion against UI truncation, collapsed newlines, odd whitespace characters, and partial formatting. It also supports sourcing the content directly from a tracked repository file (preferred) or a remote URL as fallbacks.

## When To Use
Run this workflow whenever you have a new batch of topic ideas (features, refactors, docs tasks, etc.) authored in ChatGPT (or any editor) that you want decomposed into discrete GitHub issues with automatic labeling & de‑duplication.

## Trigger
Manual dispatch only (`workflow_dispatch`). Navigate to the Actions tab → select **Agents 63 Issue Intake** → "Run workflow", then choose `chatgpt_sync` for the `intake_mode` input before providing a topic source.

## Inputs Overview (Precedence)
The workflow accepts up to three alternative sources for the topic list. Exactly one should normally be used; if multiple are provided the following precedence (highest wins) is applied:
1. `source` (preferred)
2. `raw_input`
3. `source_url`

| Input | Type | Recommended | Description |
|-------|------|-------------|-------------|
| `source` | path (string) | Yes | Path (relative to repo root) of a committed text file containing the enumerated topics (e.g. `docs/backlog/new_topics.txt`). Eliminates UI length & formatting limits. |
| `raw_input` | multiline text | Limited | Paste raw ChatGPT output directly. Subject to GitHub UI truncation (~1024 bytes); use only for short lists. |
| `source_url` | URL | Fallback | HTTP(S) URL returning plain text. Used if you have an externally hosted list. |
| `debug` | boolean (`true`/`false`) | Optional | Enables verbose diagnostics & artifact uploads for troubleshooting. |
| `intake_mode` | choice | Required | Select `chatgpt_sync` to enable topic ingestion; other options (like `agent_bridge`) target different intake flows. |

### Why `source` First?
Using a committed file via the `source` input: (a) avoids 1024‑byte truncation on the dispatch form, (b) preserves original newlines & indentation, (c) gives you a durable history of backlog drops, and (d) enables peer review via PR before issue fan‑out.

## Content Format Expectations
The strict parser targets enumerated lists, e.g.
```
1. Improve CLI help text
2. Add blended Sharpe/Sortino score
3. Refactor export layer for multi‑period
```
Acceptable numbering tokens include typical digit+dot or digit + ) patterns. Additional heuristic reconstruction attempts to restore lost newlines where the UI collapsed them.

## Normalization Pipeline
Regardless of source, the workflow routes text through a single Python normalizer (`decode_raw_input.py`) that:
1. Removes BOM, zero‑width, NBSP, and normalizes CRLF → LF.
2. Replaces tabs with single spaces.
3. Heuristically reinserts newlines before recognized enumerator tokens if the input arrived collapsed.
4. Records counts of each normalization action.
5. Emits a diagnostics JSON (`decode_debug.json`) summarizing raw length, rebuilt length, enumerator counts, transforms applied, and which source won.

## Parsing & Fallback Logic
1. Normalized text is passed to `parse_chatgpt_topics.py` (strict parser). Exit codes:
   * 0: Success (topics.json written)
   * 2 / 4: Structural or empty errors → workflow fails
   * 3: No enumerators detected → potential formatting collapse
2. If exit code 3 and enumerators were inferred by heuristics, the fallback splitter (`fallback_split.py`) generates a minimal `topics.json` using best‑effort segmentation.
3. The chosen topics list is then fed to the issue sync step.

## Issue Synchronization
The GitHub Script step:
- Computes normalized labels & deterministic colors.
- Performs fuzzy (Levenshtein) similarity against open & closed issues to reuse or reopen rather than duplicate.
- Creates new issues only when no sufficiently similar match exists.
- Applies any standard labels (defined in the script) and leaves a breadcrumb comment on reused issues.

## Artifacts (when `debug=true`)
| Artifact | Purpose |
|----------|---------|
| `raw_input.txt` | Raw captured text (if `raw_input` used). |
| `input_repo_file.txt` | Exact file contents ingested (if `source` points to a repository file). |
| `input_url.txt` | Remote body snapshot (if `source_url`). |
| `decode_debug.json` | Full normalization + reconstruction diagnostics. |
| `topics.json` | Final parsed (or fallback) topic structures. |
| `fallback_topics.json` | Produced only when fallback splitter triggered. |

## Typical Workflow Examples
### A. Using a Repository File (Recommended)
1. Add a new file with enumerated topics: `docs/backlog/2025-09-ideas.txt`.
2. Commit & push the file on a branch; open a PR if you want review.
3. Merge (or use the branch path) so the file exists on the default branch you will dispatch from.
4. Run the workflow specifying:
   - intake_mode: `chatgpt_sync`
   - source: `docs/backlog/2025-09-ideas.txt`
   - debug: `false` (or `true` if first run)
5. Inspect run summary; verify topics created/reused.

### B. Quick Small Paste
1. Copy up to ~15 short lines from ChatGPT output.
2. Run workflow with only `raw_input` filled.
3. If you see `raw_len = 1024` in diagnostics, your text was truncated—switch to `source`.

### C. Remote URL
1. Host plain text (e.g. a Gist raw URL).
2. Provide `source_url` and leave other source inputs blank.
3. Run with `debug=true` first time to verify encoding & enumerator counts.

## Interpreting Diagnostics
Key fields in `decode_debug.json`:
- `source_used`: Which of source/raw_input/source_url won.
- `raw_len` vs `rebuilt_len`: Large delta often means newline reconstruction occurred.
- `enumerator_count` / `distinct_enumerators`: Sanity check enumeration coverage.
- `applied_transforms`: List of normalization steps invoked.
- `fallback_used`: true if strict parser failed (exit 3) and fallback splitter generated topics.

## Failure Modes & Remedies
| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `raw_len` exactly 1024 | UI truncation | Use `source`. |
| Exit code 3, fallback triggered | Collapsed or malformed numbering | Review fallback segmentation; consider revising source file. |
| Zero topics parsed (exit 4) | No enumerators | Add numbering tokens (`1.`, `2.`) to list. |
| Duplicate issues not created | Title similarity threshold reused old issue | Edit existing issue title/content if a new variant required. |

## Operational Guidelines
1. Prefer `source` for any non-trivial list (>8 lines or when formatting matters).
2. Keep topic files in a dated directory (`docs/backlog/`).
3. Use PR review to refine wording before fan‑out.
4. Run with `debug=true` on first introduction of a new formatting style.
5. Archive or delete stale backlog files after issues are created to avoid confusion.

## Updating / Extending
If you modify parsing heuristics or add inputs:
1. Update this document (change log at bottom).
2. Adjust `decode_raw_input.py` & ensure new fields appear in `decode_debug.json`.
3. Add or adapt tests (future enhancement) for enumerator detection & normalization.
4. Consider adding mutual exclusivity validation if more parallel sources emerge.

## Change Log
- 2026-02-21: Consolidated ChatGPT sync into `agents-63-issue-intake.yml`; documentation updated for the `intake_mode` input and `source` field rename.
- 2025-09-20: Initial documentation added. Unified normalization, repo_file precedence, fallback splitter description.

---
Questions? Open an issue labeled `documentation` or `workflow`.
