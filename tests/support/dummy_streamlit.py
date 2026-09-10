"""One shared Streamlit stub for page-level unit tests.

Four page test modules each carried their own ``DummyStreamlit`` class
(``tests/app/test_data_page.py``, ``tests/app/test_results_page.py``,
``tests/app/test_validation_page_renders.py``, ``tests/streamlit/test_mc_page.py``).
Widget API drift in one copy could not fail the others until runtime, so this
module merges their capabilities into a single class.

Two conventions keep the merge safe:

* **Recording is additive.** Every widget method records its call. A page test
  only sees the attributes its own fixture binds onto the fake ``streamlit``
  module, so extra methods here cannot change what a page can reach.
* **Recorder aliases share one list.** The original copies spelled the same
  recorder differently (``captions`` vs ``caption_messages``, ``metrics`` vs
  ``metric_calls``). Both names point at the same list object here, so either
  spelling observes the same calls.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


class RecordingSessionState(dict[str, Any]):
    """``st.session_state`` double that records which keys were read.

    ``tests/app/test_validation_page_renders.py`` asserts both that
    ``returns_df`` was read and that ``app_data`` was NOT, so ``get`` must stay
    the only recording path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.get_keys: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.get_keys.append(key)
        return super().get(key, default)


class StubContext:
    """Context manager returned by ``columns``/``expander``/``tabs``/``sidebar``.

    Entering yields the stub itself and attribute access proxies to it, so
    ``with cols[0]:`` and ``cols[0].metric(...)`` both land on the same
    recorders as a direct ``st.metric(...)`` call.
    """

    def __init__(self, stub: "DummyStreamlit") -> None:
        self._stub = stub

    def __enter__(self) -> "DummyStreamlit":
        return self._stub

    def __exit__(self, *_exc: object) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stub, name)


class ProgressBar:
    """Return value of ``st.progress`` / ``Placeholder.progress``."""

    def __init__(self, stub: "DummyStreamlit") -> None:
        self._stub = stub

    def progress(self, value: float, text: str | None = None) -> None:
        self._stub.progress_calls.append((value, text))


class Placeholder:
    """Return value of ``st.empty()``."""

    def __init__(self, stub: "DummyStreamlit") -> None:
        self._stub = stub

    def progress(self, value: float, text: str | None = None) -> ProgressBar:
        bar = ProgressBar(self._stub)
        bar.progress(value, text=text)
        return bar

    def metric(self, label: str, value: Any) -> None:
        self._stub.metric_calls.append((label, value))

    def empty(self) -> None:
        return None


class ColumnConfig:
    """``st.column_config`` namespace used by the Data page's data editor."""

    class CheckboxColumn:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    class TextColumn:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None


class DummyStreamlit:
    """Merged Streamlit stub for page unit tests."""

    def __init__(self) -> None:
        self.session_state = RecordingSessionState()

        # --- recorders (aliases share one list object) -------------------
        self.title_calls: list[str] = []
        self.header_calls: list[str] = []
        self.subheaders: list[str] = []
        self.write_calls: list[str] = []
        self.write_messages = self.write_calls
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.error_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.captions: list[str] = []
        self.caption_messages = self.captions
        self.markdowns: list[str] = []
        self.markdown_messages = self.markdowns
        self.code_blocks: list[tuple[str, str | None]] = []
        self.dataframes: list[pd.DataFrame] = []
        self.metrics: list[tuple[str, Any]] = []
        self.metric_calls = self.metrics
        self.tab_groups: list[list[str]] = []
        self.tabs_calls = self.tab_groups
        self.expander_labels: list[str] = []
        self.checkbox_labels: list[str] = []
        self.button_labels: list[str] = []
        self.page_config_calls: list[dict[str, Any]] = []
        self.page_links: list[tuple[str, str]] = []
        self.downloads: list[dict[str, Any]] = []
        self.plotly_calls: list[dict[str, Any]] = []
        self.altair_payloads: list[Any] = []
        self.progress_calls: list[tuple[float, str | None]] = []
        self.selectbox_calls: list[tuple[str, list[Any]]] = []
        self.multiselect_calls: list[tuple[str, list[Any]]] = []
        self.slider_calls: list[tuple[str, dict[str, Any]]] = []

        # --- scripted responses ------------------------------------------
        self.radio_value = "Sample dataset"
        self.selectbox_value: Any = None
        self.selectbox_map: dict[str, Any] = {}
        self.selectbox_returns: list[Any] = []
        self.multiselect_returns: list[list[Any]] = []
        self.slider_returns: list[int] = []
        self.text_input_returns: list[str] = []
        self.button_responses: list[bool] = []
        self.uploaded: Any = None

        # --- state flags --------------------------------------------------
        self.rerun_called = False
        self.stop_called = False
        self.stop_error: str | None = "streamlit.stop() was called"

        self.column_config = ColumnConfig()
        self.sidebar = StubContext(self)

    # --- text -------------------------------------------------------------
    def title(self, text: str = "") -> None:
        self.title_calls.append(text)

    def header(self, text: str = "", *_args: Any, **_kwargs: Any) -> None:
        self.header_calls.append(text)

    def subheader(self, text: str = "", *_args: Any, **_kwargs: Any) -> None:
        self.subheaders.append(text)

    def write(self, text: Any = "", *_args: Any, **_kwargs: Any) -> None:
        self.write_calls.append(str(text))

    def markdown(self, text: Any = "", *_args: Any, **_kwargs: Any) -> None:
        self.markdowns.append(str(text))

    def caption(self, text: Any = "", *_args: Any, **_kwargs: Any) -> None:
        self.captions.append(str(text))

    def code(
        self, text: Any = "", *_args: Any, language: str | None = None, **_kwargs: Any
    ) -> None:
        self.code_blocks.append((str(text), language))

    def divider(self) -> None:
        return None

    def json(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    # --- status messages ---------------------------------------------------
    def info(self, message: Any = "") -> None:
        self.info_messages.append(str(message))

    def success(self, message: Any = "") -> None:
        self.success_messages.append(str(message))

    def error(self, message: Any = "") -> None:
        self.error_messages.append(str(message))

    def warning(self, message: Any = "") -> None:
        self.warning_messages.append(str(message))

    # --- inputs ------------------------------------------------------------
    def radio(self, *args: Any, **kwargs: Any) -> str:
        options = kwargs.get("options") or args[1]
        index = kwargs.get("index", 0)
        if index >= len(options):
            raise IndexError("radio index out of range")
        return self.radio_value

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label", "")
        options = kwargs.get("options")
        if options is None and len(args) > 1:
            options = args[1]
        options = list(options) if options is not None else []
        self.selectbox_calls.append((label, options))
        if label in self.selectbox_map:
            return self.selectbox_map[label]
        if self.selectbox_returns:
            return self.selectbox_returns.pop(0)
        if self.selectbox_value is not None:
            return self.selectbox_value
        index = kwargs.get("index", args[2] if len(args) > 2 else 0)
        if index is None:
            index = 0
        return options[index] if options else None

    def multiselect(self, label: str = "", options: list[Any] | None = None, **_kwargs: Any):
        self.multiselect_calls.append((label, list(options or [])))
        if self.multiselect_returns:
            return self.multiselect_returns.pop(0)
        return []

    def slider(self, label: str = "", **kwargs: Any) -> int:
        self.slider_calls.append((label, dict(kwargs)))
        if self.slider_returns:
            return int(self.slider_returns.pop(0))
        return int(kwargs.get("value", 0))

    def text_input(self, _label: str = "", value: str = "", **_kwargs: Any) -> str:
        if self.text_input_returns:
            return self.text_input_returns.pop(0)
        return value

    def button(self, label: str = "", *_args: Any, **_kwargs: Any) -> bool:
        self.button_labels.append(label)
        if self.button_responses:
            return self.button_responses.pop(0)
        return False

    def checkbox(
        self, label: str = "", value: bool = False, *, key: str | None = None, **_kwargs: Any
    ) -> bool:
        self.checkbox_labels.append(label)
        if key is None:
            return bool(value)
        if key not in self.session_state:
            self.session_state[key] = value
        return bool(self.session_state.get(key))

    def file_uploader(self, *_args: Any, **_kwargs: Any) -> Any:
        return self.uploaded

    def data_editor(self, df: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
        return df

    # --- layout ------------------------------------------------------------
    def columns(self, spec: int | list[Any]) -> list[StubContext]:
        count = spec if isinstance(spec, int) else len(spec)
        return [StubContext(self) for _ in range(count)]

    def container(self, *_args: Any, **_kwargs: Any) -> StubContext:
        return StubContext(self)

    def expander(self, label: Any = None, *_args: Any, **_kwargs: Any) -> StubContext:
        if isinstance(label, str):
            self.expander_labels.append(label)
        return StubContext(self)

    def tabs(self, labels: list[str]) -> list[StubContext]:
        self.tab_groups.append(list(labels))
        return [StubContext(self) for _ in labels]

    def spinner(self, *_args: Any, **_kwargs: Any) -> StubContext:
        return StubContext(self)

    def empty(self) -> Placeholder:
        return Placeholder(self)

    def set_page_config(self, **kwargs: Any) -> None:
        self.page_config_calls.append(kwargs)

    # --- output ------------------------------------------------------------
    def dataframe(self, df: pd.DataFrame, **_kwargs: Any) -> None:
        self.dataframes.append(df)

    def metric(self, label: str, value: Any = None, **_kwargs: Any) -> None:
        self.metrics.append((label, value))

    def altair_chart(self, payload: Any = None, **_kwargs: Any) -> None:
        self.altair_payloads.append(payload)

    def plotly_chart(self, *_args: Any, **kwargs: Any) -> None:
        self.plotly_calls.append(dict(kwargs))

    def progress(self, value: float, text: str | None = None) -> ProgressBar:
        bar = ProgressBar(self)
        bar.progress(value, text=text)
        return bar

    def download_button(self, **kwargs: Any) -> None:
        self.downloads.append(kwargs)

    def page_link(self, path: str, *, label: str, **_kwargs: Any) -> None:
        self.page_links.append((path, label))

    # --- control flow ------------------------------------------------------
    def rerun(self) -> None:
        self.rerun_called = True

    def stop(self) -> None:
        """Abort the script the way real ``st.stop()`` does.

        Real Streamlit raises, and page control flow depends on that, so this
        raises too. ``stop_error`` lets a page fixture supply its own message
        (or ``None`` to record without raising).
        """
        self.stop_called = True
        if self.stop_error is not None:
            raise AssertionError(self.stop_error)

    # --- misc --------------------------------------------------------------
    def cache_data(
        self, *_args: Any, **_kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator
