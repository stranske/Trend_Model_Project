"""App behavior baseline kit (TMP pilot).

A systematic, scenario-driven layer for:
  * wiring checks    -- does each input control actually change output?
  * sensibility      -- do economic invariants hold, and do parameter changes
                        move results in the economically expected direction?
  * regression       -- golden-master baselines that future versions diff against.

This package is intentionally self-contained so the generic parts can later be
extracted into a shared, pip-installable pytest plugin used across apps.
See ``tests/baseline/README.md`` for the full design and how to bless baselines.
"""
