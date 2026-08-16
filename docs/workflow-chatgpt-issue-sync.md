# ChatGPT Issue Sync via Agents Issue Intake

_Last updated: 2026-08-16_

Use the local `agents-issue-intake.yml` workflow to turn tracked topic files
into consistently labeled GitHub issues. The local workflow is the consumer
entry point; the implementation remains in `stranske/Workflows`.

## When to use it

Use ChatGPT sync for a reviewed batch of feature, refactor, or documentation
topics that should become discrete issues. Keep the source files in the
repository so the exact input is reviewable and reproducible.

## Run the workflow

1. Commit the topic file or files on the ref you will dispatch.
2. Open **Actions -> Agents Issue Intake -> Run workflow**.
3. Set `mode` to `chatgpt_sync`.
4. Set `topic_files` to the tracked path or supported file pattern, such as
   `topics.json` or `agents/*.md`.
5. Enable `apply_langchain_formatting` only when the batch should use that
   optional formatting pass.
6. Review the run summary and the created or reused issues.

The consumer dispatch exposes only these ChatGPT-sync inputs:

| Input | Type | Purpose |
| --- | --- | --- |
| `mode` | choice | Select `chatgpt_sync` rather than `agent_bridge`. |
| `topic_files` | string | Tracked topic path or supported file pattern. |
| `apply_langchain_formatting` | boolean | Optionally format created issues. |

Do not call the source repository's numbered reusable workflow directly from
this consumer. Do not document source-only inputs as if they were available in
the local Actions form.

## Topic quality

Each topic should contain a clear title, concrete scope, and testable acceptance
criteria. Prefer a reviewed file over a long paste so formatting and provenance
remain durable. The issue-format conventions are documented in
[`docs/ci/ISSUE_FORMAT_GUIDE.md`](ci/ISSUE_FORMAT_GUIDE.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Sync job skipped | Confirm `mode` is `chatgpt_sync`. |
| Topic file missing | Confirm the path exists on the dispatched ref. |
| Unexpected issue formatting | Recheck `apply_langchain_formatting` and the source file. |
| Duplicate issue reused | Compare the existing issue with the submitted topic before retrying. |

For shared parsing or synchronization defects, repair the canonical Workflows
implementation and deliver the consumer update through the managed sync path.
