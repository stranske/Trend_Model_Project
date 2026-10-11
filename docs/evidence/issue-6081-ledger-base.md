# Issue 6081: ledger base migration proof

Source checkout: `f9761590f9db9e6c5f3b9be161d2b07490a45dc9`.
Only 22 existing `.agents/issue-*-ledger.yml` files change, each exactly one line:
`base: phase-3` → `base: main`. This evidence document is the only non-ledger addition.
All other bytes, including task IDs/status, branches, comments and formatting, are preserved.

## Executed verification

Interpreter: Python 3.12 from the existing isolated proposed-tool environment,
with PyYAML 6.0.3. Run at repository root with `PYTHONPATH=src`:

1. `python scripts/ledger_migrate_base.py --check --default main`: exit **1**, reporting
   stale bases for issues 4848, 4849, 4850, 4851, 4868, 4871, 4872, 4876, 4881,
   4891, 4903, 4905, 4907, 4909, 4911, 4913, 4916, 5033, 5264, 5491, 5644, 5658.
2. `python scripts/ledger_migrate_base.py --default main`: exit **0**, migrated all 22.
   The YAML serializer also reflowed four files. Those incidental changes were
   removed before validation; exact original bytes were preserved apart from the base line.
3. `python scripts/ledger_migrate_base.py --check --default main`: exit **0**,
   `All ledgers already track the default branch.`
4. A second `ledger_migrate_base.main(['--default', 'main'])`: exit **0**,
   `Ledgers already matched the default branch; no updates written.`
   Every changed ledger remained byte-identical before/after this second migration.

An independent receipt harness extracted **every original ledger** from the exact
source commit into a disposable directory and invoked the unchanged helper's
`main(['--check', '--default', 'main'])`: **RED exit 1**. On the working tree the
same helper invocation returned **GREEN exit 0**. Per-file byte checks proved
`fixed == original.replace(b'base: phase-3\n', b'base: main\n', 1)` for all 22 files;
no task/branch/status content was reconstructed or rewritten.

Full original/fixed SHA256 bindings, stdout, argument vectors and exit codes are
retained in the owning lane's `evidence/20261011-6081-proof.json`, SHA256
`7442b68a87a03777a72f25990df3a0affb4514b5ac4404bdad65409a89119dd2`.
This is bounded metadata proof, **not** a full repository suite or an observed
hosted worker cycle. The system Anaconda interpreter failed on unrelated
NumPy/SciPy imports before the helper ran; that failure is not counted as RED.

## Delivery boundary

The issue 6080 worker's unchanged ledger-base preflight must be observed after
this merge. Its Hypothesis dependency repair and Maint52/71 generation remain
separate campaign follow-through, not claimed complete by this metadata PR.
