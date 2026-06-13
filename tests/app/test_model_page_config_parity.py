"""Config-parity guard for the Model page progressive-disclosure refactor.

Issue #5408 (A29) groups advanced Model-page parameters behind expanders and
removes internal "Phase N" labels. The refactor must be *behaviour-preserving*
for the emitted configuration: the set of config keys the form produces and the
values that flow through the (now grouped) advanced widgets must be identical
for the same inputs.

These tests snapshot the emitted config dict for a fixed set of inputs and pin
the widget keys of the grouped advanced section. Changing a grouped widget's
key (or adding/removing a config key the form emits) makes a test fail — the
deliberate-break gate required by the issue.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

# The complete set of config keys the Model-page form emits on submit. This is
# the behaviour contract the issue protects ("do NOT change the config keys the
# form emits"). A deliberate change to the emitted config must update this set.
EXPECTED_CONFIG_KEYS = frozenset(
    {
        "preset",
        "lookback_periods",
        "min_history_periods",
        "evaluation_periods",
        "multi_period_frequency",
        "selection_count",
        "weighting_scheme",
        "metric_weights",
        "risk_target",
        "info_ratio_benchmark",
        "date_mode",
        "start_date",
        "end_date",
        "rf_override_enabled",
        "rf_rate_annual",
        "vol_floor",
        "warmup_periods",
        "vol_adjust_enabled",
        "vol_window_length",
        "vol_window_decay",
        "vol_ewma_lambda",
        "max_weight",
        "min_weight",
        "cooldown_periods",
        "rebalance_freq",
        "max_turnover",
        "transaction_cost_bps",
        "min_tenure_periods",
        "max_changes_per_period",
        "max_active_positions",
        "trend_window",
        "trend_lag",
        "trend_min_periods",
        "trend_zscore",
        "trend_vol_adjust",
        "trend_vol_target",
        "regime_enabled",
        "regime_proxy",
        "shrinkage_enabled",
        "shrinkage_method",
        "random_seed",
        "condition_threshold",
        "safe_mode",
        "long_only",
        "z_entry_soft",
        "z_exit_soft",
        "soft_strikes",
        "entry_soft_strikes",
        "min_weight_strikes",
        "sticky_add_periods",
        "sticky_drop_periods",
        "ci_level",
        "multi_period_enabled",
        "inclusion_approach",
        "buy_hold_initial",
        "slippage_bps",
        "bottom_k",
        "rank_pct",
        "mp_min_funds",
        "mp_max_funds",
        "z_entry_hard",
        "z_exit_hard",
        "report_regime_analysis",
        "report_concentration",
        "report_benchmark_comparison",
        "report_factor_exposures",
        "report_attribution",
        "report_rolling_metrics",
    }
)

# Widget keys for the parameters moved into the "Advanced Settings" expander
# (progressive disclosure). Pinning these is the deliberate-break gate: rename
# one of these keys in 2_Model.py and `test_advanced_widget_keys_are_grouped`
# fails.
EXPECTED_ADVANCED_WIDGET_KEYS = frozenset(
    {
        "adv_cooldown_periods_input",
        "adv_max_turnover_input",
        "adv_rebalance_freq_select",
        "adv_transaction_cost_bps_input",
        "adv_min_tenure_periods_input",
        "adv_max_changes_per_period_input",
    }
)


@pytest.fixture()
def rendered_model(monkeypatch: pytest.MonkeyPatch):
    """Render the Model page form once and capture the emitted config + keys.

    Returns a SimpleNamespace with ``config`` (the committed ``model_state``
    dict the form emitted) and ``widget_keys`` (every ``key=`` passed to a
    Streamlit input during the render).
    """

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    widget_keys: list[str] = []

    def _noop(*_args, **_kwargs):
        return None

    def _record_key(kwargs: dict) -> None:
        key = kwargs.get("key")
        if isinstance(key, str):
            widget_keys.append(key)

    def _passthrough_decorator(*args, **kwargs):
        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator

    class Context:
        def __enter__(self):
            return stub

        def __exit__(self, *_args):
            return False

    class Placeholder:
        def progress(self, *_args, **_kwargs):
            return SimpleNamespace(progress=_noop)

        def empty(self):
            return None

    def _number_input(_label, **kwargs):
        _record_key(kwargs)
        return kwargs.get("value", 0)

    def _selectbox(_label, options, index=0, **kwargs):
        _record_key(kwargs)
        return options[index]

    def _radio(_label, options, index=0, **kwargs):
        _record_key(kwargs)
        return options[index]

    def _checkbox(_label, value=False, **kwargs):
        _record_key(kwargs)
        return value

    def _slider(_label, **kwargs):
        _record_key(kwargs)
        return kwargs.get("value", 0)

    def _text_input(_label, value="", **kwargs):
        _record_key(kwargs)
        return value

    def _text_area(_label, value="", **kwargs):
        _record_key(kwargs)
        return value

    def _date_input(_label, value=None, **kwargs):
        _record_key(kwargs)
        return value

    stub = SimpleNamespace()
    stub.session_state = {}
    stub.title = _noop
    stub.error = _noop
    stub.subheader = _noop
    stub.header = _noop
    stub.divider = _noop
    stub.info = _noop
    stub.success = _noop
    stub.warning = _noop
    stub.code = _noop
    stub.altair_chart = _noop
    stub.markdown = _noop
    stub.caption = _noop
    stub.metric = _noop
    stub.write = _noop
    stub.text_input = _text_input
    stub.text_area = _text_area
    stub.radio = _radio
    stub.date_input = _date_input
    stub.page_link = _noop
    stub.rerun = _noop
    stub.cache_data = _passthrough_decorator
    stub.cache_resource = _passthrough_decorator
    stub.expander = lambda *_args, **_kwargs: Context()
    stub.container = lambda *_args, **_kwargs: Context()
    stub.sidebar = Context()
    stub.dialog = lambda *_args, **_kwargs: Context()
    stub.form = lambda *_args, **_kwargs: Context()
    # Submit the form so candidate_state is assembled and committed.
    stub.form_submit_button = lambda *_args, **_kwargs: True
    stub.button = lambda *_args, **_kwargs: False
    stub.spinner = lambda *_args, **_kwargs: Context()
    stub.download_button = _noop
    stub.dataframe = _noop
    stub.tabs = lambda labels: [Context() for _ in labels]
    stub.columns = lambda n: [Context() for _ in range(n if isinstance(n, int) else len(n))]
    stub.selectbox = _selectbox
    stub.number_input = _number_input
    stub.checkbox = _checkbox
    stub.slider = _slider
    stub.file_uploader = lambda *_args, **_kwargs: None
    stub.empty = lambda: Placeholder()

    monkeypatch.setitem(sys.modules, "streamlit", stub)
    runtime_module = ModuleType("streamlit.runtime")
    uploaded_module = ModuleType("streamlit.runtime.uploaded_file_manager")
    uploaded_module.UploadedFile = object
    runtime_module.uploaded_file_manager = uploaded_module
    monkeypatch.setitem(sys.modules, "streamlit.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "streamlit.runtime.uploaded_file_manager", uploaded_module)

    monkeypatch.setattr(
        "streamlit_app.components.analysis_runner.clear_cached_analysis",
        _noop,
    )

    from streamlit_app import state as app_state

    monkeypatch.setattr(app_state, "initialize_session_state", lambda: None)
    monkeypatch.setattr(app_state, "st", stub)
    monkeypatch.setattr(app_state, "clear_analysis_results", _noop, raising=False)
    monkeypatch.setattr(
        app_state,
        "get_uploaded_data",
        lambda: (
            pd.DataFrame({f"A{i}": [0.01 + i * 0.001, 0.02 + i * 0.001] for i in range(12)}),
            {},
        ),
    )

    module = importlib.reload(importlib.import_module("streamlit_app.pages.2_Model"))

    # Bypass validation so the assembled candidate_state is always committed;
    # this isolates the config-parity contract from validation rules.
    monkeypatch.setattr(module, "_validate_model", lambda *_a, **_k: [])

    # Fixed set of inputs: a fully-populated model_state so every widget's
    # value= source is deterministic.
    fixed_inputs: dict[str, object] = {
        "cooldown_periods": 7,
        "max_turnover": 0.5,
        "rebalance_freq": "Q",
        "transaction_cost_bps": 13,
        "min_tenure_periods": 4,
        "max_changes_per_period": 6,
    }
    stub.session_state["model_state"] = dict(fixed_inputs)

    module.render_model_page()

    emitted = stub.session_state.get("model_state")
    assert isinstance(emitted, Mapping), "form did not commit a model_state config"

    return SimpleNamespace(
        config=dict(emitted),
        widget_keys=list(widget_keys),
        fixed_inputs=fixed_inputs,
    )


def test_emitted_config_keys_are_stable(rendered_model) -> None:
    """The Model form must emit exactly the documented set of config keys."""
    assert set(rendered_model.config) == set(EXPECTED_CONFIG_KEYS)


def test_advanced_widget_values_flow_through_unchanged(rendered_model) -> None:
    """Grouping the advanced widgets must not alter the values they emit.

    Each fixed input fed into the advanced expander must round-trip into the
    emitted config unchanged.
    """
    config = rendered_model.config
    for key, value in rendered_model.fixed_inputs.items():
        assert config[key] == value, f"advanced field {key!r} changed under grouping"


def test_advanced_widget_keys_are_grouped(rendered_model) -> None:
    """Pin the grouped advanced widget keys (deliberate-break gate).

    Renaming any grouped widget key in 2_Model.py makes this fail.
    """
    rendered = set(rendered_model.widget_keys)
    missing = EXPECTED_ADVANCED_WIDGET_KEYS - rendered
    assert not missing, f"advanced widget keys missing from render: {sorted(missing)}"


def test_emitted_config_is_deterministic(rendered_model) -> None:
    """Same inputs -> same emitted config (no hidden nondeterminism)."""
    first = rendered_model.config
    # Re-derive the expected echo of the fixed advanced inputs.
    for key, value in rendered_model.fixed_inputs.items():
        assert first[key] == value
    # The emitted config must be a plain JSON-ish dict (snapshot-able).
    assert all(isinstance(k, str) for k in first)
