"""Coverage tests for work package t12: ds_streamlit, validation, ui_ingest, patch, ui_mapping."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError as PydValidationError

from trend_analysis.config import patch as pm
from trend_analysis.config import ui_mapping as um
from trend_analysis.config import validation as v
from trend_analysis.io import ui_ingest as ui
from trend_analysis.io.date_correction import DateCorrectionResult
from trend_analysis.io.market_data import MarketDataValidationError

_root = Path(__file__).resolve().parent.parent
_REPO_ROOT = (
    _root if (_root / "design-system" / "ds_streamlit.py").is_file() else _root.parent / "tree"
)


class _FakeStreamlit(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.markdown_calls: list[tuple[Any, ...]] = []
        self.button_result = False
        self.button_calls: list[tuple[Any, ...]] = []
        self.expander_calls: list[tuple[Any, ...]] = []

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        self.markdown_calls.append(args)

    def button(self, *args: Any, **kwargs: Any) -> bool:
        self.button_calls.append(args)
        return self.button_result

    def expander(self, *args: Any, **kwargs: Any) -> Any:
        self.expander_calls.append((args, kwargs))
        parent = self

        class _Ctx:
            def __enter__(self) -> Any:
                return parent

            def __exit__(self, *exc: Any) -> bool:
                return False

        return _Ctx()


@pytest.fixture()
def ds(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    spec = importlib.util.spec_from_file_location(
        "ds_streamlit_wp12", _REPO_ROOT / "design-system" / "ds_streamlit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._fake_st = fake  # type: ignore[attr-defined]
    return module


def test_empty_state_without_cta_escapes_and_returns_false(ds: Any) -> None:
    result = ds.empty_state("<b>Hi</b>", "a & b")
    assert result is False
    html = ds._fake_st.markdown_calls[0][0]
    assert "&lt;b&gt;Hi&lt;/b&gt;" in html
    assert "a &amp; b" in html
    assert ds._fake_st.button_calls == []


def test_empty_state_with_cta_clicked_invokes_callback(ds: Any) -> None:
    ds._fake_st.button_result = True
    seen: list[str] = []
    result = ds.empty_state("T", "D", cta_label="Go", on_click=lambda: seen.append("x"))
    assert result is True
    assert seen == ["x"]
    assert ds._fake_st.button_calls[0][0] == "Go"


def test_empty_state_with_cta_not_clicked(ds: Any) -> None:
    ds._fake_st.button_result = False
    seen: list[str] = []
    result = ds.empty_state("T", "D", cta_label="Go", on_click=lambda: seen.append("x"))
    assert result is False
    assert seen == []


def test_notice_known_and_unknown_kind(ds: Any) -> None:
    ds.notice("error", title="Bad", body="oops", action="Fix it")
    html = ds._fake_st.markdown_calls[-1][0]
    assert html == (
        "<div class='ds-notice' style='background:#fef2f2;border-color:#dc262633'>"
        "<span class='ic' style='color:#dc2626'>✕</span>"
        "<div><strong>Bad</strong><br>oops<div style='margin-top:6px'>Fix it</div></div></div>"
    )
    ds.notice("bogus", body="hello")
    html2 = ds._fake_st.markdown_calls[-1][0]
    assert html2 == (
        "<div class='ds-notice' style='background:#f4f3ff;border-color:#4f46e533'>"
        "<span class='ic' style='color:#4f46e5'>i</span>"
        "<div>hello</div></div>"
    )


def test_error_delegates_to_notice(ds: Any) -> None:
    ds.error("broken", remediation="retry")
    html = ds._fake_st.markdown_calls[-1][0]
    assert "broken" in html
    assert "retry" in html


def test_translate_error_branches(ds: Any) -> None:
    assert ds.translate_error(ValueError("financing_mode missing")) == (
        "Financing mode isn't set for this run.",
        "Choose a financing mode (e.g. per-path) and run again.",
    )
    assert ds.translate_error(ValueError("exceeds total capital"))[0] == (
        "The capital allocation isn't feasible."
    )
    assert ds.translate_error(ValueError("capital buffer too small"))[0] == (
        "The capital allocation isn't feasible."
    )
    assert ds.translate_error(ValueError("no investable funds"))[0] == (
        "No funds matched the selection filters."
    )
    assert ds.translate_error(ValueError("no_funds here"))[0] == (
        "No funds matched the selection filters."
    )
    assert ds.translate_error(RuntimeError("weird")) == (
        "Something went wrong running this step.",
        "Adjust the inputs and try again; if it persists, check the run logs.",
    )


def test_dev_note_logs(ds: Any, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="ds"):
        ds.dev_note("hello note")
    assert "hello note" in caplog.text


def test_diagnostics_expander_yields(ds: Any) -> None:
    with ds.diagnostics_expander("Diag", expanded=True) as ctx:
        assert ctx is None
    assert ds._fake_st.expander_calls == [(("Diag",), {"expanded": True})]


def test_availability_badge_plain_and_html(ds: Any) -> None:
    assert ds.availability_badge("  Ready  ") == " · Ready"
    assert ds.availability_badge("<x>", plain=False) == ("<span class='ds-badge'>&lt;x&gt;</span>")


def test_humanize_id_mapping_and_segments(ds: Any) -> None:
    assert ds.humanize_id("a:b", mapping={"a:b": "Label"}) == "Label"
    assert ds.humanize_id("ns:my_fund-name") == "my fund name"
    assert ds.humanize_id("550e8400-e29b-41d4-a716-446655440000") == "item"
    assert ds.humanize_id("deadbeefcafe") == "item"


def test_human_label_segment_filters(ds: Any) -> None:
    assert ds._human_label_segment("my_fund") == "my fund"
    assert ds._human_label_segment("550e8400-e29b-41d4-a716-446655440000") == ""
    assert ds._human_label_segment("123_456") == ""
    assert ds._human_label_segment("fund_1234abcd") == "fund"
    assert ds._human_label_segment("") == ""


def test_is_opaque_segment_and_token(ds: Any) -> None:
    assert ds._is_opaque_id_segment("") is True
    assert ds._is_opaque_id_segment("550e8400-e29b-41d4-a716-446655440000") is True
    assert ds._is_opaque_id_segment("deadbeef") is True
    assert ds._is_opaque_id_segment("myfund") is False
    assert ds._is_opaque_id_token("deadbeef") is True
    assert ds._is_opaque_id_token("ab12") is True
    assert ds._is_opaque_id_token("abcd") is False
    assert ds._is_opaque_id_token("ab") is False


def _err(validator: str, schema: dict[str, Any] | None = None, value: Any = None) -> Any:
    class _E:
        pass

    e = _E()
    e.validator = validator
    e.schema = schema or {}
    e.validator_value = value
    e.message = "msg"
    e.instance = None
    e.absolute_path = []
    return e


def test_validation_result_coerce_none_and_strings() -> None:
    r = v.ValidationResult(errors=None)  # type: ignore[arg-type]
    assert r.errors == []
    r2 = v.ValidationResult(errors=["boom"])  # type: ignore[arg-type]
    assert r2.errors[0].path == "<root>"
    assert r2.errors[0].message == "boom"
    assert r2.valid is False
    assert v.ValidationResult._coerce_issue_strings("notalist") == "notalist"


def test_validate_config_non_mapping() -> None:
    result = v.validate_config([1, 2])  # type: ignore[arg-type]
    assert result.valid is False
    assert result.errors[0].message == "Configuration payload must be an object."
    assert result.errors[0].actual == "list"


def test_suggest_additional_property_branches() -> None:
    e1 = _err("additionalProperties", {})
    assert v._suggest_additional_property(None, e1) is None
    e2 = _err("additionalProperties", {})
    assert v._suggest_additional_property("foo", e2) == (
        "Remove 'foo' or move it under the correct section."
    )
    e3 = _err("additionalProperties", {"properties": {"version": {}, "data": {}}})
    out = v._suggest_additional_property("verson", e3)
    assert out is not None and "Valid keys:" in out


def test_expected_for_error_branches() -> None:
    assert v._expected_for_error(_err("minimum", {"minimum": 5})) == ">= 5"
    assert v._expected_for_error(_err("maximum", {"maximum": 9})) == "<= 9"
    assert v._expected_for_error(_err("minItems", {"minItems": 2})) == "at least 2 items"
    assert v._expected_for_error(_err("maxItems", {"maxItems": 3})) == "at most 3 items"
    assert v._expected_for_error(_err("pattern", {"pattern": "^a$"})) == "match pattern ^a$"
    assert v._expected_for_error(_err("pattern", {})) == "matching value"


def test_suggestion_for_error_branches() -> None:
    assert v._suggestion_for_error(_err("minimum", {"minimum": 5})) == "Use a value >= 5."
    assert v._suggestion_for_error(_err("maximum", {"maximum": 9})) == "Use a value <= 9."
    assert v._suggestion_for_error(_err("minItems", {"minItems": 2})) == (
        "Provide at least 2 entries."
    )
    assert v._suggestion_for_error(_err("maxItems", {"maxItems": 3})) == (
        "Provide no more than 3 entries."
    )
    assert v._suggestion_for_error(_err("type", {"type": "string"})) == ("Provide a string value.")


def test_check_required_sections_non_mapping() -> None:
    errors: list[Any] = []
    cfg = {"version": 1, "data": "nope", "portfolio": {}}
    v._check_required_sections(cfg, errors)
    msgs = {(e.path, e.message) for e in errors}
    assert ("data", "Section must be an object.") in msgs


def test_check_version_field_non_string() -> None:
    errors: list[Any] = []
    v._check_version_field({"version": 3}, errors)
    assert len(errors) == 1
    assert errors[0].message == "Version must be a string."
    assert errors[0].actual == "int"


def test_require_field_blank_value() -> None:
    errors: list[Any] = []
    v._require_field(
        errors,
        {"date_column": "  "},
        "data",
        "date_column",
        expected="non-empty string",
        suggestion="Set it.",
    )
    assert len(errors) == 1
    assert errors[0].path == "data.date_column"
    assert errors[0].actual == "  "


def test_collect_feasibility_and_coerce_float() -> None:
    assert v._coerce_float("not-a-number") is None
    assert v._coerce_float(None) is None
    assert v._coerce_float(True) is None
    assert v._coerce_float("2.5") == 2.5
    errs = v.collect_feasibility_errors(
        {"portfolio": {"constraints": {"max_weight": 0.5, "min_weight": 0.8}}}
    )
    assert len(errs) == 1
    assert errs[0].message == "max_weight is below min_weight."


def test_feasibility_unknown_capacity_returns_silently(tmp_path: Path) -> None:
    errs = v.collect_feasibility_errors(
        {
            "portfolio": {
                "long_only": True,
                "constraints": {"max_weight": 0.01, "min_weight": 0.0},
            },
            "data": {"csv_path": "missing.csv"},
        },
        base_path=tmp_path,
    )
    assert errs == []
    assert (
        v._effective_holding_count(
            {"data": {"csv_path": "missing.csv"}, "portfolio": {}}, {}, tmp_path
        )
        is None
    )


def test_portfolio_selection_rank_non_mapping() -> None:
    errors: list[Any] = []
    v._check_portfolio_selection_requirements(
        {"portfolio": {"selection_mode": "rank", "rank": "oops"}}, errors
    )
    assert len(errors) == 1
    assert errors[0].message == "Rank settings must be an object."
    assert errors[0].actual == "str"


def test_manual_selection_empty_list() -> None:
    errors: list[Any] = []
    v._check_manual_selection_requirements(
        {"portfolio": {"selection_mode": "manual", "manual_list": []}}, errors
    )
    assert len(errors) == 1
    assert errors[0].message == "Manual list cannot be empty."
    assert errors[0].actual == "empty list"


def test_rank_value_ranges_bool_ignored() -> None:
    errors: list[Any] = []
    v._check_rank_value_ranges(
        {"portfolio": {"rank": {"inclusion_approach": "top_n", "n": True}}}, errors
    )
    assert errors == []
    errors2: list[Any] = []
    v._check_rank_value_ranges(
        {"portfolio": {"rank": {"inclusion_approach": "top_pct", "pct": True}}}, errors2
    )
    assert errors2 == []
    errors3: list[Any] = []
    v._check_rank_value_ranges(
        {"portfolio": {"rank": {"inclusion_approach": "top_n", "n": 0}}}, errors3
    )
    assert len(errors3) == 1
    assert errors3[0].message == "top_n must be a positive integer."


def test_count_available_funds_csv_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "broken.csv"
    broken.write_text("not,really,csv", encoding="utf-8")

    def boom(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("nrows") == 0:
            raise OSError("cannot read header")
        return pd.read_csv(*args, **kwargs)

    monkeypatch.setattr(v.pd, "read_csv", boom)
    assert v._count_available_funds({"data": {"csv_path": "broken.csv"}}, tmp_path) is None


def test_count_available_funds_branches(tmp_path: Path) -> None:
    assert v._count_available_funds({}, tmp_path) is None
    assert v._count_available_funds({"data": {}}, tmp_path) is None
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00bad")
    # Unreadable/odd CSV still returns a count or None; force exception via directory
    assert v._count_available_funds({"data": {"csv_path": "missing.csv"}}, tmp_path) is None
    good = tmp_path / "funds.csv"
    good.write_text("Date,FundA,FundB,RF\n2024-01-31,0.1,0.2,0.0\n", encoding="utf-8")
    assert (
        v._count_available_funds(
            {"data": {"csv_path": "funds.csv", "risk_free_column": "RF"}}, tmp_path
        )
        == 2
    )


def test_error_from_exception_empty_and_actual_helpers() -> None:
    assert v._error_from_exception(ValueError("   "), {}) is None
    issue = v._error_from_exception(ValueError("portfolio.rank: bad"), {"portfolio": {"rank": 7}})
    assert issue is not None
    assert issue.path == "portfolio.rank"
    assert issue.actual == 7
    assert v._actual_from_path({"a": 1}, "<root>") is None
    assert v._actual_from_path({"a": 1}, "") is None
    assert v._actual_from_path({"a": [10, 20]}, "a[1]") == 20
    assert v._actual_from_path({"a": [10]}, "a[5]") is None
    assert v._actual_from_path({"a": 1}, "b.c") is None
    assert v._actual_from_path({"a": {"b": 3}}, "a.b") == 3


def test_format_helpers() -> None:
    assert v._format_path([0, "a"]) == "[0].a"
    assert v._format_path(["a", 1]) == "a[1]"
    assert v._format_path([]) == "<root>"
    assert v._format_actual(None) == "null"
    assert v._format_actual("missing") == "missing"
    assert v._format_actual("x") == '"x"'


def test_append_issue_dedupes() -> None:
    bucket: list[Any] = []
    issue = v.ConfigIssue(path="p", message="m")
    v._append_issue(bucket, issue)
    v._append_issue(bucket, v.ConfigIssue(path="p", message="m"))
    assert len(bucket) == 1
    v._append_issue(bucket, v.ConfigIssue(path="p", message="other"))
    assert len(bucket) == 2


def test_effective_holding_count_fallback(tmp_path: Path) -> None:
    good = tmp_path / "f.csv"
    good.write_text("Date,A,B\n2024-01-31,1,2\n", encoding="utf-8")
    cfg = {"data": {"csv_path": "f.csv"}, "portfolio": {}}
    assert v._effective_holding_count(cfg, {}, tmp_path) == 2


def test_normalise_header_value() -> None:
    assert ui._normalise_header_value(None) == ""
    assert ui._normalise_header_value("\ufeffDate") == "Date"
    assert ui._normalise_header_value(float("nan")) == ""
    assert ui._normalise_header_value(42) == "42"
    assert ui._normalise_header_value(" Fund ") == " Fund "


def test_extract_headers_csv_only_whitespace_line() -> None:
    assert ui._extract_headers_from_bytes(b"\n", is_excel=False) == []
    assert ui._extract_headers_from_bytes(b"\r\n", is_excel=False) == []


def test_extract_headers_csv_and_empty() -> None:
    assert ui._extract_headers_from_bytes(b"", is_excel=False) == []
    assert ui._extract_headers_from_bytes(b"", is_excel=True) == []
    assert ui._extract_headers_from_bytes(b"Date,FundA\n2024-01-31,1\n", is_excel=False) == [
        "Date",
        "FundA",
    ]
    assert ui._extract_headers_from_bytes(b"\xef\xbb\xbfDate,X\n", is_excel=False) == ["Date", "X"]


def test_extract_headers_csv_stop_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptyReader:
        def __iter__(self) -> Any:
            return self

        def __next__(self) -> Any:
            raise StopIteration

    monkeypatch.setattr(ui.csv, "reader", lambda *a, **k: _EmptyReader())
    assert ui._extract_headers_from_bytes(b"ignored\n", is_excel=False) == []


def test_extract_headers_excel(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("no engine")

    monkeypatch.setattr(ui.pd, "read_excel", boom)
    assert ui._extract_headers_from_bytes(b"data", is_excel=True) is None
    empty = pd.DataFrame()
    monkeypatch.setattr(ui.pd, "read_excel", lambda *a, **k: empty)
    assert ui._extract_headers_from_bytes(b"data", is_excel=True) == []
    one = pd.DataFrame([["Date", 5]])
    monkeypatch.setattr(ui.pd, "read_excel", lambda *a, **k: one)
    assert ui._extract_headers_from_bytes(b"data", is_excel=True) == ["Date", "5"]


def test_apply_original_headers() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    values = df.to_numpy().copy()
    ui._apply_original_headers(df, None)
    assert list(df.columns) == ["a", "b"]
    assert df.to_numpy().tolist() == values.tolist()
    ui._apply_original_headers(df, ["only"])
    assert list(df.columns) == ["a", "b"]
    ui._apply_original_headers(df, ["X", "Y"])
    assert list(df.columns) == ["X", "Y"]
    assert df.to_numpy().tolist() == [[1, 2]]


def test_allocate_unique_name() -> None:
    occ: set[str] = set()
    assert ui._allocate_unique_name("Fund", occ) == "Fund"
    assert ui._allocate_unique_name("Fund", occ) == "Fund_2"
    assert ui._allocate_unique_name("Fund", occ) == "Fund_3"
    assert occ == {"Fund", "Fund_2", "Fund_3"}


def test_sanitize_formula_headers() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["=FundA", "+FundA", "Date"])
    out = ui._sanitize_formula_headers(df)
    assert list(out.columns) == ["FundA", "FundA_2", "Date"]
    df2 = pd.DataFrame([[1]], columns=["=Date"])
    out2 = ui._sanitize_formula_headers(df2)
    assert list(out2.columns) == ["Date"]
    df3 = pd.DataFrame([[1]], columns=["@"])
    out3 = ui._sanitize_formula_headers(df3)
    assert list(out3.columns) == ["safe_column"]
    df4 = pd.DataFrame([[1]], columns=["Safe"])
    out4 = ui._sanitize_formula_headers(df4)
    assert out4 is df4


def test_find_date_column_miss() -> None:
    df = pd.DataFrame({"X": [1]})
    assert ui._find_date_column(df) is None
    df2 = pd.DataFrame({" date ": [1]})
    assert ui._find_date_column(df2) == " date "


def test_raise_date_issue_fixable_preview() -> None:
    from trend_analysis.io.date_correction import DateCorrection

    corr = DateCorrection(
        row_index=0,
        original_value="2024-02-30",
        corrected_value="2024-02-29",
        explanation="fixed",
    )
    res = DateCorrectionResult(
        corrections=[corr],
        unfixable=[],
        trailing_empty_rows=[],
        droppable_empty_rows=[],
    )
    with pytest.raises(MarketDataValidationError) as exc:
        ui._raise_date_issue(res, auto_fix_dates=False)
    assert exc.value.issues == ["1 row(s) can be auto-corrected."]


def test_raise_date_issue_messages() -> None:
    res = DateCorrectionResult(
        corrections=[], unfixable=[(0, "bad")], trailing_empty_rows=[], droppable_empty_rows=[]
    )
    with pytest.raises(MarketDataValidationError) as e1:
        ui._raise_date_issue(res, auto_fix_dates=True)
    assert str(e1.value) == "Date column contains values that cannot be auto-corrected."
    assert e1.value.issues == ["Unfixable dates detected (row 1: 'bad')."]
    with pytest.raises(MarketDataValidationError) as e2:
        ui._raise_date_issue(res, auto_fix_dates=False)
    assert str(e2.value) == (
        "Date column has fixable issues. Re-run with --auto-fix-dates to apply them."
    )
    clean = DateCorrectionResult(
        corrections=[], unfixable=[], trailing_empty_rows=[], droppable_empty_rows=[]
    )
    with pytest.raises(MarketDataValidationError):
        ui._raise_date_issue(clean, auto_fix_dates=False)


def test_inspect_and_load_missing_date(tmp_path: Path) -> None:
    p = tmp_path / "nodate.csv"
    p.write_text("A,B\n1,2\n", encoding="utf-8")
    with pytest.raises(MarketDataValidationError, match="Missing a 'Date' column"):
        ui.inspect_ui_date_issues(p)
    with pytest.raises(MarketDataValidationError, match="Missing a 'Date' column"):
        ui.load_ui_dataset(p)


def test_load_ui_dataset_happy_and_rename(tmp_path: Path) -> None:
    p = tmp_path / "ok.csv"
    p.write_text("date,FundA\n2024-01-31,0.01\n2024-02-29,0.02\n", encoding="utf-8")
    frame, meta, summary = ui.load_ui_dataset(p)
    assert list(frame.columns) == ["FundA"]
    assert frame["FundA"].tolist() == [0.01, 0.02]
    assert frame.index.tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert summary.corrected_dates == 0
    assert summary.dropped_rows == 0
    assert meta.frequency == "M"


def test_load_ui_dataset_bad_dates_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("Date,FundA\nnot-a-date,0.01\n2024-02-29,0.02\n", encoding="utf-8")
    with pytest.raises(
        MarketDataValidationError,
        match="Date column contains values that cannot be auto-corrected.",
    ) as excinfo:
        ui.load_ui_dataset(p, auto_fix_dates=True)
    assert excinfo.value.issues == ["Unfixable dates detected (row 1: 'not-a-date')."]


def test_load_ui_dataset_fixable_without_autofix(tmp_path: Path) -> None:
    p = tmp_path / "fixable.csv"
    p.write_text("Date,FundA\n2024-02-30,0.01\n2024-03-31,0.02\n", encoding="utf-8")
    with pytest.raises(
        MarketDataValidationError,
        match="Date column has fixable issues. Re-run with --auto-fix-dates to apply them.",
    ):
        ui.load_ui_dataset(p, auto_fix_dates=False)


def test_inspect_ui_date_issues_excel(tmp_path: Path) -> None:
    p = tmp_path / "book.xlsx"
    pd.DataFrame({"Date": ["2024-01-31"], "FundA": [0.01]}).to_excel(p, index=False)
    result = ui.inspect_ui_date_issues(p)
    assert result.unfixable == []


def test_load_ui_dataset_excel(tmp_path: Path) -> None:
    p = tmp_path / "book.xlsx"
    pd.DataFrame({"Date": ["2024-01-31", "2024-02-29"], "FundA": [0.01, 0.02]}).to_excel(
        p, index=False
    )
    frame, meta, summary = ui.load_ui_dataset(p)
    assert list(frame.columns) == ["FundA"]
    assert frame["FundA"].tolist() == [0.01, 0.02]
    assert summary.corrected_dates == 0
    assert meta.mode.value == "returns"


def _op(op: str, path: str, **kw: Any) -> Any:
    return pm.PatchOperation(op=op, path=path, **kw)


def test_patch_operation_root_pointer_rejected() -> None:
    with pytest.raises(PydValidationError, match="at least one segment"):
        _op("set", "/", value=1)


def test_apply_and_validate() -> None:
    patch = pm.ConfigPatch(operations=[_op("set", "version", value="2")], summary="bump")
    updated, result = pm.apply_and_validate({"version": "1"}, patch)
    assert updated["version"] == "2"
    assert hasattr(result, "valid")


def test_parse_dotpath_invalid_and_format() -> None:
    with pytest.raises(ValueError, match="dotpath"):
        pm._parse_dotpath("a..b")
    assert pm._format_dotpath([0, "a"]) == "[0].a"
    assert pm._format_dotpath(["a", 1]) == "a[1]"


def test_resolve_parent_branches() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        pm._resolve_parent({}, [], "set")
    with pytest.raises(TypeError, match="requires an object"):
        pm._resolve_parent({"a": 5}, ["a", "b", "c"], "set")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="Did you mean"):
        pm._resolve_parent({"versio": "1"}, ["version", "x"], "set")
    with pytest.raises(TypeError, match="requires a list"):
        pm._resolve_parent({"a": {}}, ["a", 0, "b"], "set")
    with pytest.raises(IndexError, match="non-negative"):
        pm._resolve_parent({"a": [1]}, ["a", -1, "b"], "set")
    root: dict[str, Any] = {"a": [None]}
    parent, leaf = pm._resolve_parent(root, ["a", 0, "b"], "remove", allow_missing=True)
    assert parent is None and leaf is None
    root2: dict[str, Any] = {}
    parent2, leaf2 = pm._resolve_parent(root2, ["gone", "x"], "remove", allow_missing=True)
    assert parent2 is None and leaf2 is None
    parent_oob, leaf_oob = pm._resolve_parent(
        {"a": [1]}, ["a", 9, "b"], "remove", allow_missing=True
    )
    assert parent_oob is None and leaf_oob is None
    with pytest.raises(KeyError, match="does not exist"):
        pm._resolve_parent({"a": [1]}, ["a", 9, "b"], "remove")


def test_apply_operation_branches() -> None:
    with pytest.raises(TypeError, match="remove requires a list"):
        pm._apply_operation({}, 0, _op("remove", "a[0]"))
    with pytest.raises(TypeError, match="object container"):
        pm._apply_operation([1, 2], "x", _op("remove", "x"))
    lst = [1, 2, 3]
    pm._apply_operation(lst, 1, _op("remove", "a[1]"))
    assert lst == [1, 3]
    d: dict[str, Any] = {"x": 1}
    pm._apply_operation(d, "x", _op("remove", "x"))
    assert d == {}
    with pytest.raises(TypeError, match="set requires a list"):
        pm._apply_operation({}, 0, _op("set", "a[0]", value=9))
    lst2 = [1]
    pm._apply_operation(lst2, 3, _op("set", "a[3]", value=9))
    assert lst2 == [1, None, None, 9]
    with pytest.raises(TypeError, match="set requires an object"):
        pm._apply_operation([], "x", _op("set", "x", value=9))
    set_neg = pm.PatchOperation.model_construct(op="set", path="a[0]", value=9)
    with pytest.raises(IndexError, match="non-negative"):
        pm._apply_operation([1], -1, set_neg)
    with pytest.raises(TypeError, match="not an index"):
        pm._apply_operation({}, 0, _op("append", "a[0]", value=9))
    with pytest.raises(TypeError, match="append requires an object"):
        pm._apply_operation([], "x", _op("append", "x", value=9))
    with pytest.raises(TypeError, match="merge requires a list"):
        pm._apply_operation({}, 0, _op("merge", "a[0]", value={}))
    lst3: list[Any] = [None]
    pm._apply_operation(lst3, 0, _op("merge", "a[0]", value={"k": 1}))
    assert lst3 == [{"k": 1}]
    with pytest.raises(TypeError, match="merge requires an object container"):
        pm._apply_operation([], "x", _op("merge", "x", value={}))
    d2: dict[str, Any] = {}
    pm._apply_operation(d2, "cfg", _op("merge", "cfg", value={"a": 1}))
    assert d2 == {"cfg": {"a": 1}}
    with pytest.raises(TypeError, match="merge requires an object at"):
        pm._apply_operation({"cfg": 5}, "cfg", _op("merge", "cfg", value={"a": 1}))
    merge_neg = pm.PatchOperation.model_construct(op="merge", path="a[0]", value={"k": 1})
    with pytest.raises(IndexError, match="non-negative"):
        pm._apply_operation([{}], -1, merge_neg)
    lst_extend: list[Any] = []
    pm._apply_operation(lst_extend, 2, _op("merge", "a[2]", value={"k": 1}))
    assert lst_extend == [None, None, {"k": 1}]
    bad_merge_val = pm.PatchOperation.model_construct(op="merge", path="a", value=5)
    with pytest.raises(TypeError, match="dict value"):
        pm._apply_operation({"a": {}}, "a", bad_merge_val)
    with pytest.raises(KeyError, match="does not exist"):
        pm._apply_operation({"versio": 1}, "version", _op("set", "version", value=2))
    append_leaf = pm.PatchOperation.model_construct(op="append", path="items", value=1)
    with pytest.raises(TypeError, match="not an index"):
        pm._apply_operation({}, 0, append_leaf)
    merge_list_leaf = pm.PatchOperation.model_construct(op="merge", path="a[0]", value={"k": 1})
    with pytest.raises(TypeError, match="merge requires a list"):
        pm._apply_operation({}, 0, merge_list_leaf)
    bogus = pm.PatchOperation.model_construct(op="bogus", path="x", value=1)
    with pytest.raises(ValueError, match="unsupported operation"):
        pm._apply_operation({}, "x", bogus)


def test_collect_paths_and_deep_merge() -> None:
    assert pm._collect_paths({"a": [{"b": 1}]}) == ["a", "a[0]", "a[0].b"]
    assert pm._collect_paths([1, {"x": 2}]) == ["[0]", "[1]", "[1].x"]
    target: dict[str, Any] = {"a": {"x": 1, "y": 2}}
    pm._deep_merge(target, {"a": {"y": 3, "z": 4}})
    assert target == {"a": {"x": 1, "y": 3, "z": 4}}


def test_apply_patch_empty_segments_via_construct() -> None:
    op = pm.PatchOperation.model_construct(op="set", path="", value=1)
    patch = pm.ConfigPatch.model_construct(
        operations=[op], needs_review=False, risk_flags=[], summary="s"
    )
    with pytest.raises(ValueError, match="at least one segment"):
        pm.apply_patch({}, patch)


def test_build_sample_split_empty_and_explicit_missing() -> None:
    with pytest.raises(ValueError, match="Dataset is empty"):
        um.build_sample_split(pd.DatetimeIndex([]), {})
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    with pytest.raises(ValueError, match="Explicit date mode requires"):
        um.build_sample_split(idx, {"date_mode": "explicit", "start_date": "2021-01-01"})


def test_build_sample_split_explicit_ok_and_clamp() -> None:
    idx = pd.date_range("2020-01-31", periods=48, freq="ME")
    out = um.build_sample_split(
        idx,
        {
            "date_mode": "explicit",
            "start_date": "2020-01-15",
            "end_date": "2023-12-15",
            "multi_period_frequency": "A",
            "lookback_periods": 2,
        },
    )
    assert out["out_start"] == "2020-01"
    assert out["in_start"] <= out["in_end"]


def test_build_sample_split_explicit_bad_dates_fallthrough() -> None:
    idx = pd.date_range("2020-01-31", periods=48, freq="ME")
    out = um.build_sample_split(
        idx, {"date_mode": "explicit", "start_date": "garbage!!", "end_date": "also-bad"}
    )
    assert out == {
        "in_start": "2020-01",
        "in_end": "2022-12",
        "out_start": "2023-01",
        "out_end": "2023-12",
    }


def test_build_signals_config_branches() -> None:
    base = um.build_signals_config({})
    assert "min_periods" not in base and "vol_target" not in base
    assert um.build_signals_config({"min_periods": "7", "window": 10})["min_periods"] == 7
    assert "min_periods" not in um.build_signals_config({"min_periods": "abc"})
    assert "min_periods" not in um.build_signals_config({"min_periods": [1]})
    assert "min_periods" not in um.build_signals_config({"min_periods": 0})
    assert um.build_signals_config({"min_periods": 99, "window": 10})["min_periods"] == 10
    assert "vol_target" not in um.build_signals_config({"vol_target": "abc", "vol_adjust": True})
    assert "vol_target" not in um.build_signals_config({"vol_target": -1, "vol_adjust": True})
    assert um.build_signals_config({"vol_target": 0.2, "vol_adjust": True})["vol_target"] == 0.2


def test_normalise_metric_weights_branches() -> None:
    out = um.normalise_metric_weights(
        {"sharpe": 2.0, "unknown": 5.0, "vol": "bad", "info_ratio": -1}
    )
    assert out == {"sharpe": 1.0}
    assert um.normalise_metric_weights({}) == {
        "sharpe": 1.0 / 3,
        "return_ann": 1.0 / 3,
        "drawdown": 1.0 / 3,
    }


def test_build_portfolio_config_branches() -> None:
    w = {"sharpe": 1.0}
    cfg = um.build_portfolio_config(
        {
            "inclusion_approach": "buy_and_hold",
            "buy_hold_initial": "random",
            "bottom_k": 3,
            "max_changes_per_period": 2,
            "max_active_positions": 5,
        },
        w,
    )
    assert cfg["selection_mode"] == "buy_and_hold"
    assert cfg["rank"]["bottom_k"] == 3
    assert cfg["turnover_budget_max_changes"] == 2
    assert cfg["constraints"]["max_active_positions"] == 5
    cfg2 = um.build_portfolio_config({}, w)
    assert cfg2["selection_mode"] == "rank"
    assert "bottom_k" not in cfg2["rank"]


def test_build_config_from_ui_state_variants() -> None:
    returns = pd.DataFrame(
        {"FundA": [0.01, 0.02, 0.03]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    state: dict[str, Any] = {
        "metric_weights": {"sharpe": 1.0},
        "vol_window_decay": "bogus",
        "vol_ewma_lambda": 5.0,
        "random_seed": "not-an-int",
        "z_entry_hard": 2.0,
        "z_exit_hard": -2.0,
        "missing_limit": 3,
        "risk_free_column": " RF ",
    }
    cfg = um.build_config_from_ui_state(
        returns=returns,
        model_state=state,
        benchmark=None,
        frequency="monthly",
        csv_path=None,
    )
    dumped = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)  # type: ignore[arg-type]
    assert dumped["data"]["missing_limit"] == 3
    assert dumped["data"]["risk_free_column"] == "RF"
    assert dumped["portfolio"]["threshold_hold"]["z_entry_hard"] == 2.0
    assert dumped["portfolio"]["threshold_hold"]["z_exit_hard"] == -2.0
    assert "vol_window_decay" not in dumped.get("signals", {})
    assert dumped["portfolio"]["policy"] == "threshold_hold"


def test_build_config_multi_period_without_user_dates() -> None:
    returns = pd.DataFrame(
        {"FundA": [0.01] * 36},
        index=pd.date_range("2020-01-31", periods=36, freq="ME"),
    )
    cfg = um.build_config_from_ui_state(
        returns=returns,
        model_state={"multi_period_enabled": True},
        benchmark=None,
        frequency="monthly",
        csv_path=None,
    )
    dumped = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)  # type: ignore[arg-type]
    assert dumped["multi_period"] == {
        "frequency": "A",
        "in_sample_len": 3,
        "out_sample_len": 1,
        "min_history_periods": 3,
        "start": "2020-01-31",
        "end": "2022-12-31",
        "start_mode": "in",
    }


def test_build_config_from_ui_state_non_datetime_and_multiperiod() -> None:
    returns = pd.DataFrame({"FundA": [0.01] * 24}, index=pd.RangeIndex(24))
    returns.index = pd.date_range("2022-01-31", periods=24, freq="ME")
    plain = pd.DataFrame({"FundA": [0.01, 0.02]}, index=[0, 1])
    plain.index = pd.Index(["2024-01-31", "2024-02-29"])
    cfg = um.build_config_from_ui_state(
        returns=plain,
        model_state={"multi_period_enabled": True, "start_date": "junk!!", "end_date": "junk!!"},
        benchmark=None,
        frequency="monthly",
        csv_path=None,
    )
    dumped = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)  # type: ignore[arg-type]
    assert dumped["multi_period"] == {
        "frequency": "A",
        "in_sample_len": 3,
        "out_sample_len": 1,
        "min_history_periods": 3,
        "start": "2024-01-31",
        "end": "2024-02-29",
        "start_mode": "in",
    }
