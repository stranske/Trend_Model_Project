from __future__ import annotations

import pandas as pd

from trend_analysis.ui import rank_widgets


class DummyResult:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value
        self.diagnostic = None

    def __bool__(self) -> bool:
        return True


def _build_sample_csv(tmp_path) -> str:
    df = pd.DataFrame(
        {
            "Date": [
                "2020-01-31",
                "2020-02-29",
                "2020-03-31",
                "2020-04-30",
                "2020-05-31",
                "2020-06-30",
            ],
            "FundA": [0.01, 0.02, 0.03, 0.01, 0.02, 0.01],
            "FundB": [0.02, 0.01, 0.00, 0.02, 0.03, 0.01],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def _unpack_step1(ui):
    step1_box = ui.children[0]
    return step1_box.children


def _unpack_rank_controls(ui):
    (
        _step1_box,
        mode_dd,
        random_n_int,
        vol_ck,
        target_vol,
        use_rank_ck,
        next_btn_1,
        rank_box,
        manual_box,
        out_fmt,
        run_btn,
        output,
    ) = ui.children
    return {
        "mode_dd": mode_dd,
        "random_n_int": random_n_int,
        "vol_ck": vol_ck,
        "target_vol": target_vol,
        "use_rank_ck": use_rank_ck,
        "next_btn_1": next_btn_1,
        "rank_box": rank_box,
        "manual_box": manual_box,
        "out_fmt": out_fmt,
        "run_btn": run_btn,
        "output": output,
    }


def test_rank_widgets_loads_csv_and_updates_fields(tmp_path) -> None:
    ui = rank_widgets.build_ui()
    (
        source_tb,
        csv_path,
        _file_up,
        load_btn,
        _load_out,
        idx_select,
        bench_select,
        in_start,
        in_end,
        out_start,
        out_end,
    ) = _unpack_step1(ui)

    csv_path.value = _build_sample_csv(tmp_path)
    load_btn.click()

    assert source_tb.value == "Path/URL"
    assert set(idx_select.options) == {"FundA", "FundB"}
    assert set(bench_select.options) == {"FundA", "FundB"}
    assert idx_select.layout.display == "flex"
    assert bench_select.layout.display == "flex"
    assert in_start.value == "2020-01"
    assert in_end.value == "2020-03"
    assert out_start.value == "2020-04"
    assert out_end.value == "2020-06"


def test_rank_widgets_manual_flow(monkeypatch, tmp_path) -> None:
    ui = rank_widgets.build_ui()
    (
        _source_tb,
        csv_path,
        _file_up,
        load_btn,
        _load_out,
        _idx_select,
        _bench_select,
        _in_start,
        _in_end,
        _out_start,
        _out_end,
    ) = _unpack_step1(ui)
    controls = _unpack_rank_controls(ui)

    scores = pd.Series({"FundA": 0.12, "FundB": 0.08})

    def fake_run_analysis(*_args, **_kwargs):
        return DummyResult({"scores": scores})

    export_calls: list[dict[str, object]] = []

    def fake_export_data(data, prefix, formats, formatter):
        export_calls.append(
            {
                "data": data,
                "prefix": prefix,
                "formats": formats,
                "formatter": formatter,
            }
        )

    monkeypatch.setattr(rank_widgets.pipeline, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(rank_widgets.export, "make_summary_formatter", lambda *_: "fmt")
    monkeypatch.setattr(rank_widgets.export, "format_summary_text", lambda *_: "summary")
    monkeypatch.setattr(rank_widgets.export, "export_data", fake_export_data)

    csv_path.value = _build_sample_csv(tmp_path)
    load_btn.click()

    controls["mode_dd"].value = "random"
    assert controls["random_n_int"].layout.display == "flex"

    controls["vol_ck"].value = False
    assert controls["target_vol"].layout.display == "none"

    controls["mode_dd"].value = "rank"
    assert controls["use_rank_ck"].value is True

    controls["mode_dd"].value = "manual"
    assert controls["manual_box"].layout.display == "flex"

    controls["next_btn_1"].click()

    manual_rows = controls["manual_box"].children[1:]
    checkbox, weight = manual_rows[0].children
    checkbox.value = True
    weight.value = 0.6

    controls["out_fmt"].value = "csv"
    controls["run_btn"].click()

    assert export_calls
    assert export_calls[0]["formats"] == ["csv"]
    assert export_calls[0]["prefix"].startswith("IS_2020-01_OS_2020-04")
