from __future__ import annotations

import asyncio
import importlib
import importlib.util
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, cast

import pandas as pd
import yaml

from ..config import Config
from ..config.models import DEFAULTS
from ..diagnostics import coerce_pipeline_result
from .plugins import discover_plugins, iter_plugins
from .store import ParamStore
from .utils import _find_config_directory, debounce, list_builtin_cfgs

# Use _find_config_directory from utils instead of local duplicate


if TYPE_CHECKING:  # pragma: no cover
    from ..config.models import ConfigProtocol as ConfigType
else:
    from typing import Any as ConfigType

from .. import export, pipeline, weighting

_NOTEBOOK_DEP_MESSAGE = (
    "Notebook UI requires optional dependencies (ipywidgets and IPython.display). "
    "Install the 'notebook' extra or add the packages manually to enable the GUI."
)


class _WidgetModuleProxy:
    """Proxy that tolerates optional ipywidgets imports."""

    __slots__ = ("_module", "_overrides", "_error")

    def __init__(self, error_message: str) -> None:
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_overrides", {})
        object.__setattr__(self, "_error", error_message)

    def bind(self, module: Any) -> None:
        object.__setattr__(self, "_module", module)

    @property
    def is_bound(self) -> bool:
        return object.__getattribute__(self, "_module") is not None

    def __getattr__(self, name: str) -> Any:
        overrides: Dict[str, Any] = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        module = object.__getattribute__(self, "_module")
        if module is None:
            if name.startswith("__"):
                raise AttributeError(name)
            fallback = _WIDGET_FALLBACKS.get(name)
            if fallback is not None:
                return fallback
            return _MissingWidgetAttr(name, object.__getattribute__(self, "_error"))
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        overrides: Dict[str, Any] = object.__getattribute__(self, "_overrides")
        overrides[name] = value

    def clear_overrides(self, predicate: Callable[[str, Any], bool] | None = None) -> None:
        overrides: Dict[str, Any] = object.__getattribute__(self, "_overrides")
        if not overrides:
            return
        if predicate is None:
            overrides.clear()
            return
        for key in list(overrides):
            if predicate(key, overrides[key]):
                overrides.pop(key, None)


def _make_missing_callable(target: str) -> Callable[..., Any]:
    def _raiser(*_: Any, **__: Any) -> None:
        raise ImportError(
            f"Notebook UI requires {target} from IPython.display; {_NOTEBOOK_DEP_MESSAGE}"
        )

    setattr(_raiser, "__trend_stub__", True)
    return _raiser


class _MissingWidgetAttr:
    """Callable placeholder that raises when optional deps are absent."""

    __slots__ = ("_name", "_message")

    def __init__(self, name: str, message: str) -> None:
        self._name = name
        self._message = message

    def _raise(self) -> None:
        raise ImportError(f"{self._message} (missing '{self._name}')")

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - simple raiser
        self._raise()

    def __getattr__(self, _: str) -> Any:  # pragma: no cover - passthrough
        self._raise()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<_MissingWidgetAttr {self._name}>"


class _GenericWidgetStub:
    """Minimal widget stand-in that satisfies observe/click APIs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - simple shim
        value = kwargs.get("value")
        if value is None and args and not isinstance(args[0], (list, tuple)):
            value = args[0]
        children = kwargs.get("children")
        if children is None and args and isinstance(args[0], (list, tuple)):
            children = args[0]
        self.value = value
        self.options = list(kwargs.get("options", []))
        self.description = kwargs.get("description", "")
        self.children = tuple(children or ())
        self.layout = SimpleNamespace(display="", border="")
        self._observers: list[Callable[[dict[str, Any]], None]] = []
        self._click_handlers: list[Callable[[Any], None]] = []
        for key, val in kwargs.items():  # expose any additional traits
            setattr(self, key, val)

    def observe(
        self, callback: Callable[[dict[str, Any]], None], names: Any | None = None
    ) -> None:  # pragma: no cover - trivial list append
        self._observers.append(callback)

    def on_click(self, callback: Callable[[Any], None]) -> None:  # pragma: no cover
        self._click_handlers.append(callback)

    def trigger(self, value: Any) -> None:  # pragma: no cover - helper for tests
        self.value = value
        change = {"new": value}
        for cb in list(self._observers):
            cb(change)

    def click(self) -> None:  # pragma: no cover - helper for tests
        for cb in list(self._click_handlers):
            cb(self)


_WIDGET_FALLBACK_NAMES = {
    "FileUpload",
    "Dropdown",
    "Label",
    "Button",
    "VBox",
    "HBox",
    "BoundedIntText",
    "BoundedFloatText",
    "FloatText",
    "FloatSlider",
    "SelectMultiple",
    "Checkbox",
    "ToggleButtons",
    "IntSlider",
}
_WIDGET_FALLBACKS: Dict[str, type[_GenericWidgetStub]] = {
    name: _GenericWidgetStub for name in _WIDGET_FALLBACK_NAMES
}


widgets: Any = _WidgetModuleProxy(_NOTEBOOK_DEP_MESSAGE)
FileLink: Any = _make_missing_callable("FileLink")
Javascript: Any = _make_missing_callable("Javascript")
display: Any = _make_missing_callable("display")
DataGrid: Any | None = None
HAS_DATAGRID = False
_NOTEBOOK_IMPORT_FAILED = False
_AUTOLOADED_DATAGRID: Any | None = None

STATE_FILE = Path.home() / ".trend_gui_state.yml"
WEIGHT_STATE_FILE = STATE_FILE.with_suffix(".pkl")


def _load_notebook_deps() -> None:
    """Load optional notebook dependencies on demand."""

    global FileLink, Javascript, display, _NOTEBOOK_IMPORT_FAILED

    widgets_ready = not isinstance(widgets, _WidgetModuleProxy) or widgets.is_bound
    callables_ready = all(
        not getattr(obj, "__trend_stub__", False) for obj in (FileLink, Javascript, display)
    )
    if widgets_ready and callables_ready:
        if isinstance(widgets, _WidgetModuleProxy) and widgets.is_bound:
            widgets.clear_overrides(lambda _name, val: val is _GenericWidgetStub)
        _ensure_datagrid_available()
        return

    missing: list[str] = []

    if not widgets_ready:
        try:
            widgets_mod = importlib.import_module("ipywidgets")
        except ImportError:
            missing.append("ipywidgets")
        else:
            if isinstance(widgets, _WidgetModuleProxy):
                widgets.bind(widgets_mod)
                widgets.clear_overrides(lambda _name, val: val is _GenericWidgetStub)
            elif globals().get("widgets") is None:  # pragma: no cover - edge overrides
                globals()["widgets"] = widgets_mod
            widgets_ready = True

    if not callables_ready:
        try:
            display_mod = importlib.import_module("IPython.display")
        except ImportError:
            missing.append("IPython.display")
        else:
            if getattr(FileLink, "__trend_stub__", False):
                FileLink = display_mod.FileLink
            if getattr(Javascript, "__trend_stub__", False):
                Javascript = display_mod.Javascript
            if getattr(display, "__trend_stub__", False):
                display = display_mod.display
            callables_ready = True

    if missing:
        if not _NOTEBOOK_IMPORT_FAILED:
            _NOTEBOOK_IMPORT_FAILED = True
            warnings.warn(
                f"Notebook UI requires {', '.join(missing)}. {_NOTEBOOK_DEP_MESSAGE}",
                RuntimeWarning,
            )
    else:
        _NOTEBOOK_IMPORT_FAILED = False

    if widgets_ready and callables_ready:
        if isinstance(widgets, _WidgetModuleProxy) and widgets.is_bound:
            widgets.clear_overrides(lambda _name, val: val is _GenericWidgetStub)
        _ensure_datagrid_available()
        return

    _ensure_datagrid_available()


def _ensure_datagrid_available() -> None:
    """Best-effort import for optional ipydatagrid dependency."""

    global DataGrid, HAS_DATAGRID, _AUTOLOADED_DATAGRID

    try:
        datagrid_mod = importlib.import_module("ipydatagrid")
    except ImportError:
        if DataGrid is None:
            HAS_DATAGRID = False
            _AUTOLOADED_DATAGRID = None
        return

    candidate = getattr(datagrid_mod, "DataGrid", None)
    if candidate is None:
        if DataGrid is None or DataGrid is _AUTOLOADED_DATAGRID:
            DataGrid = None
            HAS_DATAGRID = False
            _AUTOLOADED_DATAGRID = None
        return

    should_replace = DataGrid is None or DataGrid is _AUTOLOADED_DATAGRID
    if should_replace:
        DataGrid = candidate
        _AUTOLOADED_DATAGRID = candidate
        try:
            setattr(DataGrid, "__trend_autoload__", True)
        except Exception:  # pragma: no cover - not all classes allow attribute set
            pass
    HAS_DATAGRID = True


def _datagrid_can_mount() -> bool:
    """Return True if ``DataGrid`` is compatible with the current widgets stack."""

    if DataGrid is None:
        return False
    widget_base = getattr(widgets, "Widget", None)
    if widget_base is None or isinstance(widget_base, _MissingWidgetAttr):
        # Either widgets are stubbed out or ipywidgets is unavailable. In those
        # environments ``widgets.VBox`` resolves to our lightweight stub, so
        # any DataGrid stand-in is acceptable.
        return True
    vbox_cls = getattr(widgets, "VBox", None)
    container_is_real = False
    if isinstance(vbox_cls, type):
        try:
            container_is_real = issubclass(vbox_cls, widget_base)
        except TypeError:
            container_is_real = False
    if not container_is_real:
        # Tests frequently patch ``widgets.VBox`` with a simple stub even when a
        # genuine ipywidgets module is available. Treat those scenarios like the
        # stubbed environment above so FakeDataGrid instances remain valid.
        return True
    try:
        return isinstance(DataGrid, type) and issubclass(DataGrid, widget_base)
    except TypeError:
        return False


# Probe availability once at import so downstream tests detect the flag.
_ensure_datagrid_available()


def load_state() -> ParamStore:
    """Load persisted GUI state from ``STATE_FILE`` if possible."""
    store = ParamStore()
    try:
        if STATE_FILE.exists():
            loaded_store = ParamStore.from_yaml(STATE_FILE)
            loaded_store.cfg = _validate_loaded_gui_config(loaded_store.cfg)
            store = loaded_store
    except Exception as exc:  # pragma: no cover - malformed file
        warnings.warn(f"Failed to load state: {exc}")
    try:
        if WEIGHT_STATE_FILE.exists():
            with WEIGHT_STATE_FILE.open("rb") as fh:
                data = pickle.load(fh)
            if not isinstance(data, dict) or "adaptive_bayes_posteriors" not in data:
                raise ValueError("weight state must use the adaptive_bayes_posteriors mapping")
            posterior_state = data["adaptive_bayes_posteriors"]
            if not isinstance(posterior_state, dict):
                raise ValueError("adaptive_bayes_posteriors must be a mapping")
            store.weight_state = posterior_state
    except Exception as exc:  # pragma: no cover - malformed file
        warnings.warn(f"Failed to load weight state: {exc}")
    return store


def save_state(store: ParamStore) -> None:
    """Persist ``store`` to ``STATE_FILE``."""
    STATE_FILE.write_text(yaml.safe_dump(store.to_dict()))
    if store.weight_state is not None:
        with WEIGHT_STATE_FILE.open("wb") as fh:
            pickle.dump({"adaptive_bayes_posteriors": store.weight_state}, fh)


def reset_weight_state(store: ParamStore) -> None:
    """Clear in-memory and persisted weighting state."""
    store.weight_state = None
    if WEIGHT_STATE_FILE.exists():  # pragma: no cover - file may not exist
        WEIGHT_STATE_FILE.unlink()


def _as_dict(v: Any) -> Dict[str, Any]:
    return dict(v) if isinstance(v, dict) else {}


def _normalized_export_formats(value: Any, *, warn_invalid: bool = False) -> list[str]:
    """Return usable export formats, preserving a safe GUI default."""

    values = list(value) if isinstance(value, (list, tuple)) else []
    formats = [
        format_name
        for format_name in values
        if isinstance(format_name, str) and format_name and format_name in export.EXPORTERS
    ]
    rejected = [format_name for format_name in values if format_name not in formats]
    if warn_invalid and rejected:
        warnings.warn(
            "Ignoring unselectable export format(s): "
            + ", ".join(repr(format_name) for format_name in rejected)
            + "; using the first registered format instead.",
            stacklevel=2,
        )
    return formats or ["xlsx"]


_REQUIRED_MAPPING_SECTIONS = (
    "data",
    "preprocessing",
    "vol_adjust",
    "sample_split",
    "portfolio",
    "benchmarks",
    "metrics",
    "export",
    "run",
)


def _ensure_mapping_sections(cfg: Dict[str, Any]) -> None:
    """Best-effort ensure mapping types for top-level config sections."""
    for section in _REQUIRED_MAPPING_SECTIONS:
        cfg[section] = _as_dict(cfg.get(section))
    if "multi_period" not in cfg:
        cfg["multi_period"] = {}
    elif cfg["multi_period"] is not None:
        cfg["multi_period"] = _as_dict(cfg.get("multi_period"))


def build_config_dict(store: ParamStore) -> Dict[str, Any]:
    """Return the config dictionary kept in ``store`` as a plain dict."""
    cfg = dict(store.cfg)

    _ensure_mapping_sections(cfg)
    return cfg


_DEFAULT_VERSION_CACHE: str | None = None


def _ensure_version(cfg: Dict[str, Any]) -> None:
    """Ensure ``cfg`` contains a ``version`` entry."""

    if "version" in cfg and isinstance(cfg["version"], str) and cfg["version"].strip():
        return

    global _DEFAULT_VERSION_CACHE
    if _DEFAULT_VERSION_CACHE is None:
        default_version: str | None = None
        try:
            with DEFAULTS.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                raw = data.get("version")
                if isinstance(raw, str) and raw.strip():
                    default_version = raw
        except Exception:  # pragma: no cover - IO failures fall back to hardcoded value
            default_version = None

        if not default_version:
            default_version = "1"
        _DEFAULT_VERSION_CACHE = default_version

    cfg.setdefault("version", _DEFAULT_VERSION_CACHE)


def build_config_from_store(store: ParamStore) -> ConfigType:
    """Convert ``store`` into a :class:`Config` object."""
    cfg: Dict[str, Any] = build_config_dict(store)
    _ensure_version(cfg)
    return Config(**cfg)


def _clear_grid_error_border(grid: Any) -> None:
    """Clear a transient edit error without assuming a notebook event loop exists."""

    def clear_border() -> None:
        grid.layout.border = ""

    try:
        asyncio.get_running_loop().call_later(1.0, clear_border)
    except RuntimeError:
        # Widget construction and tests can run outside an active asyncio loop.
        clear_border()


_RETIRED_GUI_TOP_LEVEL_KEYS = frozenset({"mode", "rank", "use_ranking", "use_vol_adjust"})


def _validate_loaded_gui_config(raw: Any) -> dict[str, Any]:
    """Reject non-canonical GUI configuration before replacing live state."""

    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    forbidden = sorted(_RETIRED_GUI_TOP_LEVEL_KEYS.intersection(raw))
    if forbidden:
        names = ", ".join(forbidden)
        raise ValueError(
            f"unsupported retired top-level configuration key(s): {names}; "
            "use portfolio.selection_mode, portfolio.rank, and vol_adjust.enabled"
        )
    export_cfg = raw.get("export")
    if not isinstance(export_cfg, dict) or "formats" not in export_cfg:
        return raw
    normalized = dict(raw)
    sanitized_export = dict(export_cfg)
    sanitized_export["formats"] = _normalized_export_formats(
        export_cfg["formats"], warn_invalid=True
    )
    normalized["export"] = sanitized_export
    return normalized


def _build_step0(
    store: ParamStore, *, on_config_replaced: Callable[[], None] | None = None
) -> widgets.Widget:
    """Return widgets for Step 0 (config loader/editor)."""

    _load_notebook_deps()
    assert widgets is not None and FileLink is not None and display is not None

    upload = widgets.FileUpload(accept=".yml", multiple=False)
    template = widgets.Dropdown(options=list_builtin_cfgs(), description="Template")
    if HAS_DATAGRID and DataGrid is not None and _datagrid_can_mount():
        grid_df = pd.DataFrame(list(store.cfg.items()), columns=["Key", "Value"])
        # ``Value`` may contain scalars of varying dtypes; keep it as ``object`` so
        # edits with strings, ints or floats don't trigger pandas' incompatible
        # dtype warnings (which will become errors in future releases).
        grid_df["Value"] = grid_df["Value"].astype(object)
        grid = DataGrid(grid_df, editable=True)

        def on_cell_change(event: dict[str, Any]) -> None:
            if event.get("column") != 1:  # value column only
                return
            key = grid_df.iloc[event["row"], 0]
            old = grid_df.iloc[event["row"], 1]
            new = event["new"]
            try:
                parsed = yaml.safe_load(new)
                candidate = dict(store.cfg)
                candidate[key] = parsed
                store.cfg = _validate_loaded_gui_config(candidate)
                grid_df.iloc[event["row"], 1] = parsed
                store.dirty = True
            except Exception:
                grid_df.iloc[event["row"], 1] = old
                grid.layout.border = "2px solid red"
                _clear_grid_error_border(grid)

        try:
            grid.on("cell_edited", on_cell_change)
        except AttributeError:
            # Handle case where DataGrid doesn't have expected API
            pass
    else:  # pragma: no cover - optional dep
        grid = widgets.Label("ipydatagrid not installed")

    save_btn = widgets.Button(description="💾 Save config")
    download_btn = widgets.Button(description="⬇️ Download")

    def refresh_grid() -> None:
        if hasattr(grid, "data"):
            with grid.hold_trait_notifications():
                grid.data = [store.cfg]

    def on_upload(change: dict[str, Any], *, store: ParamStore) -> None:
        if change["new"]:
            try:
                item = next(iter(upload.value.values()))
                loaded = yaml.safe_load(item["content"].decode("utf-8"))
                store.cfg = _validate_loaded_gui_config(loaded)
                store.dirty = True
                reset_weight_state(store)
                refresh_grid()
                if on_config_replaced is not None:
                    on_config_replaced()
            except (TypeError, ValueError, yaml.YAMLError) as exc:
                warnings.warn(f"Invalid uploaded configuration: {exc}")

    def on_template(change: dict[str, Any], *, store: ParamStore) -> None:
        name = change["new"]
        cfg_dir = _find_config_directory()
        path = cfg_dir / f"{name}.yml"
        try:
            content = path.read_text()
            store.cfg = _validate_loaded_gui_config(yaml.safe_load(content))
            store.dirty = True
            reset_weight_state(store)
            refresh_grid()
            if on_config_replaced is not None:
                on_config_replaced()
        except FileNotFoundError:
            warnings.warn(f"Template config file not found: {path}")
        except PermissionError:
            warnings.warn(f"Permission denied reading template config: {path}")
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            warnings.warn(f"Invalid YAML in template config {path}: {exc}")
        except Exception as exc:
            warnings.warn(f"Failed to load template config {path}: {exc}")

    def on_save(_: Any, *, store: ParamStore) -> None:
        save_state(store)
        store.dirty = False

    def on_download(_: Any, *, store: ParamStore) -> None:
        path = STATE_FILE.with_name("config_download.yml")
        path.write_text(yaml.safe_dump(store.to_dict()))
        cast(Any, display)(cast(Any, FileLink)(path))

    upload.observe(lambda ch, store=store: on_upload(ch, store=store), names="value")
    template.observe(lambda ch, store=store: on_template(ch, store=store), names="value")
    save_btn.on_click(lambda btn, store=store: on_save(btn, store=store))
    download_btn.on_click(lambda btn, store=store: on_download(btn, store=store))

    return widgets.VBox([template, upload, grid, widgets.HBox([save_btn, download_btn])])


def _build_rank_options(store: ParamStore) -> widgets.Widget:
    """Return widgets for ranking configuration (Step 2)."""

    _load_notebook_deps()
    assert widgets is not None
    from ..core.rank_selection import METRIC_REGISTRY

    rank_cfg = store.cfg.setdefault("portfolio", {}).setdefault("rank", {})

    incl_dd = widgets.Dropdown(
        options=["top_n", "top_pct", "threshold"],
        value=rank_cfg.get("inclusion_approach", "top_n"),
        description="Approach",
    )
    metric_dd = widgets.Dropdown(
        options=list(METRIC_REGISTRY) + ["blended"],
        value=rank_cfg.get("score_by", "Sharpe"),
        description="Score By",
    )
    n_int = widgets.BoundedIntText(value=rank_cfg.get("n", 8), min=1, description="N")
    pct_flt = widgets.BoundedFloatText(
        value=rank_cfg.get("pct", 0.10),
        min=0.01,
        max=1.0,
        step=0.01,
        description="Pct",
    )
    thresh_f = widgets.FloatText(value=rank_cfg.get("threshold", 1.0), description="Threshold")

    try:
        weights = rank_cfg.get("blended_weights", {})
        items = list(METRIC_REGISTRY)
        m1_dd = widgets.Dropdown(options=items, value=items[0], description="M1")
        w1_sl = widgets.FloatSlider(value=weights.get(items[0], 0.33), min=0, max=1, step=0.01)
        m2_dd = widgets.Dropdown(
            options=items,
            value=items[1] if len(items) > 1 else items[0],
            description="M2",
        )
        w2_sl = widgets.FloatSlider(
            value=weights.get(items[1] if len(items) > 1 else items[0], 0.33),
            min=0,
            max=1,
            step=0.01,
        )
        m3_dd = widgets.Dropdown(
            options=items,
            value=items[2] if len(items) > 2 else items[0],
            description="M3",
        )
        w3_sl = widgets.FloatSlider(
            value=weights.get(items[2] if len(items) > 2 else items[0], 0.34),
            min=0,
            max=1,
            step=0.01,
        )
        blended_box = widgets.VBox([m1_dd, w1_sl, m2_dd, w2_sl, m3_dd, w3_sl])
    except Exception:  # pragma: no cover - never raised
        blended_box = widgets.VBox()

    def _store_rank(_: Any = None, *, store: ParamStore) -> None:
        current_rank_cfg = store.cfg.setdefault("portfolio", {}).setdefault("rank", {})
        current_rank_cfg["inclusion_approach"] = incl_dd.value
        current_rank_cfg["score_by"] = metric_dd.value
        current_rank_cfg["n"] = int(n_int.value)
        current_rank_cfg["pct"] = float(pct_flt.value)
        current_rank_cfg["threshold"] = float(thresh_f.value)
        current_rank_cfg["blended_weights"] = {
            m1_dd.value: w1_sl.value,
            m2_dd.value: w2_sl.value,
            m3_dd.value: w3_sl.value,
        }
        store.dirty = True

    incl_dd.observe(lambda ch, store=store: _store_rank(ch, store=store), names="value")
    metric_dd.observe(lambda ch, store=store: _store_rank(ch, store=store), names="value")
    n_int.observe(lambda ch, store=store: _store_rank(ch, store=store), names="value")
    pct_flt.observe(lambda ch, store=store: _store_rank(ch, store=store), names="value")
    thresh_f.observe(lambda ch, store=store: _store_rank(ch, store=store), names="value")

    @debounce(300)
    def _on_blend(_: Any, *, store: ParamStore) -> None:
        _store_rank(store=store)

    m1_dd.observe(lambda ch, store=store: _on_blend(ch, store=store), names="value")
    w1_sl.observe(lambda ch, store=store: _on_blend(ch, store=store), names="value")
    m2_dd.observe(lambda ch, store=store: _on_blend(ch, store=store), names="value")
    w2_sl.observe(lambda ch, store=store: _on_blend(ch, store=store), names="value")
    m3_dd.observe(lambda ch, store=store: _on_blend(ch, store=store), names="value")
    w3_sl.observe(lambda ch, store=store: _on_blend(ch, store=store), names="value")

    metric_dd.observe(
        lambda change: blended_box.layout.__setattr__(
            "display", "flex" if change["new"] == "blended" else "none"
        ),
        names="value",
    )
    blended_box.layout.display = "none" if metric_dd.value != "blended" else "flex"

    return widgets.VBox([incl_dd, metric_dd, n_int, pct_flt, thresh_f, blended_box])


def _build_manual_override(store: ParamStore) -> widgets.Widget:
    """Return manual-selection grid or fallback."""

    _load_notebook_deps()
    assert widgets is not None
    port = store.cfg.setdefault("portfolio", {})
    weights = port.setdefault("custom_weights", {})
    manual = port.setdefault("manual_list", list(weights))

    datagrid_available = HAS_DATAGRID and DataGrid is not None and _datagrid_can_mount()

    if datagrid_available:
        assert DataGrid is not None  # mypy: guarded by datagrid_available
        rows = [
            {"Fund": f, "Include": f in manual, "Weight": float(weights.get(f, 0))}
            for f in sorted(set(manual) | set(weights))
        ]
        df = pd.DataFrame(rows, columns=["Fund", "Include", "Weight"])
        grid = DataGrid(df, editable=True)

        def _on_edit(event: dict[str, Any], *, store: ParamStore) -> None:
            fund = df.loc[event["row"], "Fund"]
            current_port = store.cfg.setdefault("portfolio", {})
            current_weights = current_port.setdefault("custom_weights", {})
            current_manual = current_port.setdefault("manual_list", list(current_weights))
            if event.get("column") == 1:  # Include
                include_val = bool(event.get("new"))
                if include_val and fund not in current_manual:
                    current_manual.append(fund)
                elif not include_val and fund in current_manual:
                    current_manual.remove(fund)
            elif event.get("column") == 2:  # Weight
                new_val = event.get("new")
                if new_val is None:
                    return
                try:
                    weight_val = float(new_val)
                    if weight_val < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    return
                current_weights[fund] = weight_val
                df.loc[event["row"], "Weight"] = weight_val
            store.dirty = True

        try:
            grid.on("cell_edited", lambda ev, store=store: _on_edit(ev, store=store))
        except AttributeError:
            # Handle case where DataGrid doesn't have expected API
            pass
        box = widgets.VBox([grid])
    else:  # pragma: no cover - optional dep
        opts = sorted(set(manual) | set(weights))
        warn = widgets.Label("ipydatagrid not installed")
        select = widgets.SelectMultiple(options=opts, value=tuple(manual))
        weight_boxes = [
            widgets.FloatText(value=float(weights.get(f, 0)), description=f) for f in opts
        ]

        def _on_select(change: dict[str, Any], *, store: ParamStore) -> None:
            current_port = store.cfg.setdefault("portfolio", {})
            current_port["manual_list"] = list(change["new"])
            store.dirty = True

        def _on_weight(change: dict[str, Any], fund: str, *, store: ParamStore) -> None:
            try:
                val = float(change["new"])
                if val < 0:
                    raise ValueError
            except Exception:
                return
            current_port = store.cfg.setdefault("portfolio", {})
            current_port.setdefault("custom_weights", {})[fund] = val
            store.dirty = True

        select.observe(lambda ch, store=store: _on_select(ch, store=store), names="value")
        for wdg in weight_boxes:
            wdg.observe(
                lambda ch, fund=wdg.description, store=store: _on_weight(ch, fund, store=store),
                names="value",
            )

        box = widgets.VBox([warn, select] + weight_boxes)

    box.layout.display = "none"
    return box


def _build_weighting_options(store: ParamStore) -> widgets.Widget:
    """Return weighting method dropdown and param sliders."""
    _load_notebook_deps()
    assert widgets is not None
    weight_cfg = store.cfg.setdefault("portfolio", {}).setdefault(
        "weighting", {"name": "equal", "params": {}}
    )
    options = {
        "equal": weighting.EqualWeight,
        "score_prop": weighting.ScorePropSimple,
        "score_prop_bayes": weighting.ScorePropBayesian,
        "adaptive_bayes": weighting.AdaptiveBayesWeighting,
    }
    for plugin in iter_plugins():
        name = getattr(plugin, "__name__", str(plugin)).lower()
        options[name] = plugin

    method_dd = widgets.Dropdown(
        options=list(options),
        value=weight_cfg.get("name", "equal"),
        description="Weighting",
    )
    params = weight_cfg.setdefault("params", {})
    hl = widgets.IntSlider(
        value=params.get("half_life", 90), min=30, max=365, description="half_life"
    )
    os_sl = widgets.FloatSlider(
        value=params.get("obs_sigma", 0.25),
        min=0.0,
        max=1.0,
        step=0.01,
        description="obs_sigma",
    )
    mw_sl = widgets.FloatSlider(
        value=params.get("max_w", 0.20),
        min=0.0,
        max=0.5,
        step=0.01,
        description="max_w",
    )
    pt_sl = widgets.FloatSlider(
        value=params.get("prior_tau", 1.0),
        min=0.0,
        max=5.0,
        step=0.1,
        description="prior_tau",
    )
    adv_box = widgets.VBox([hl, os_sl, mw_sl, pt_sl])

    def _store_weight(_: Any = None, *, store: ParamStore) -> None:
        current_weight_cfg = store.cfg.setdefault("portfolio", {}).setdefault(
            "weighting", {"name": "equal", "params": {}}
        )
        current_params = current_weight_cfg.setdefault("params", {})
        current_weight_cfg["name"] = method_dd.value
        current_params["half_life"] = int(hl.value)
        current_params["obs_sigma"] = float(os_sl.value)
        current_params["max_w"] = float(mw_sl.value)
        current_params["prior_tau"] = float(pt_sl.value)
        store.dirty = True

    method_dd.observe(lambda ch, store=store: _store_weight(ch, store=store), names="value")

    @debounce(300)
    def _on_param(_: Any, *, store: ParamStore) -> None:
        _store_weight(store=store)

    for wdg in (hl, os_sl, mw_sl, pt_sl):
        wdg.observe(lambda ch, store=store: _on_param(ch, store=store), names="value")

    def _toggle_adv(change: dict[str, Any]) -> None:
        adv_box.layout.display = "flex" if change["new"] == "adaptive_bayes" else "none"

    method_dd.observe(_toggle_adv, names="value")
    _toggle_adv({"new": method_dd.value})

    return widgets.VBox([method_dd, adv_box])


def _theme_script(theme: str) -> str:
    """Return JavaScript that applies a browser-consumed color-scheme setting."""

    color_scheme = {"light": "light", "dark": "dark", "system": "light dark"}.get(
        theme, "light dark"
    )
    return (
        "const root = document.documentElement;"
        f"root.dataset.trendTheme = '{theme}';"
        f"root.style.colorScheme = '{color_scheme}';"
    )


def launch() -> widgets.Widget:
    """Return the root widget for the Trend Model GUI."""
    _load_notebook_deps()
    assert widgets is not None and Javascript is not None
    discover_plugins()
    store = load_state()

    mode = widgets.Dropdown(
        options=["all", "random", "manual", "rank"],
        value=_as_dict(store.cfg.get("portfolio")).get("selection_mode", "all"),
        description="Mode",
    )
    vol_adj = widgets.Checkbox(
        value=_as_dict(store.cfg.get("vol_adjust")).get("enabled", True),
        description="Vol-Adj",
        indent=False,
    )
    theme = widgets.ToggleButtons(
        options=["system", "light", "dark"],
        value=store.theme,
        description="Theme",
    )
    export_state = _as_dict(store.cfg.get("export"))
    export_formats = _normalized_export_formats(export_state.get("formats"), warn_invalid=True)
    fmt_dd = widgets.Dropdown(
        options=sorted(export.EXPORTERS),
        value=str(export_formats[0]) if export_formats else "xlsx",
        description="Format",
    )
    run_btn = widgets.Button(description="Run")
    reset_btn = widgets.Button(description="↻ Reset")

    def on_theme(change: dict[str, Any], *, store: ParamStore) -> None:
        store.theme = change["new"]
        store.dirty = True
        js = cast(Any, Javascript)(_theme_script(str(change["new"])))
        cast(Any, display)(js)

    theme.observe(lambda ch, store=store: on_theme(ch, store=store), names="value")

    def on_mode(change: dict[str, Any], *, store: ParamStore) -> None:
        store.cfg.setdefault("portfolio", {})["selection_mode"] = change["new"]
        store.dirty = True

    mode.observe(lambda ch, store=store: on_mode(ch, store=store), names="value")

    def on_vol(change: dict[str, Any], *, store: ParamStore) -> None:
        store.cfg.setdefault("vol_adjust", {})["enabled"] = bool(change["new"])
        store.dirty = True

    def on_fmt(change: dict[str, Any], *, store: ParamStore) -> None:
        export_cfg = store.cfg.setdefault("export", {})
        export_cfg["formats"] = [change["new"]]
        store.dirty = True

    def on_run(_: Any, *, store: ParamStore) -> None:
        try:
            cfg = build_config_from_store(store)
            export_cfg = cfg.export or {}
            configured_formats = export_cfg.get("formats")
            requested_formats = (
                list(configured_formats)
                if isinstance(configured_formats, (list, tuple))
                else []
            )
            unsupported_formats = [
                fmt
                for fmt in requested_formats
                if not isinstance(fmt, str) or not fmt or fmt not in export.EXPORTERS
            ]
            if unsupported_formats:
                warnings.warn(
                    "Unsupported export format(s): "
                    + ", ".join(repr(fmt) for fmt in unsupported_formats)
                    + "; no files were written."
                )
                return

            metrics = pipeline.run(cfg)
            if metrics.empty:
                warnings.warn("Run produced no metrics; no files were exported.")
                return

            formats = _normalized_export_formats(configured_formats)
            path = Path(export_cfg.get("directory", ".")) / export_cfg.get("filename", "gui_output")
            data = {"metrics": metrics}
            if "xlsx" in formats:
                full_result = pipeline.run_full(cfg)
                res, diag = coerce_pipeline_result(full_result)
                if not res:
                    if diag:
                        warnings.warn(f"Pipeline aborted ({diag.reason_code}): {diag.message}")
                    return
                split = cfg.sample_split
                sheet_fmt = export.make_summary_formatter(
                    res,
                    str(split.get("in_start", "")),
                    str(split.get("in_end", "")),
                    str(split.get("out_start", "")),
                    str(split.get("out_end", "")),
                )
                data["summary"] = pd.DataFrame()
                export.export_to_excel(
                    data,
                    str(Path(path).with_suffix(".xlsx")),
                    default_sheet_formatter=sheet_fmt,
                )
                formats = [fmt for fmt in formats if fmt != "xlsx"]
            if formats:
                export.export_data(data, str(path), formats=formats)
            save_state(store)
            store.dirty = False
        except Exception as exc:
            warnings.warn(f"Run failed: {exc}")

    vol_adj.observe(lambda ch, store=store: on_vol(ch, store=store), names="value")
    fmt_dd.observe(lambda ch, store=store: on_fmt(ch, store=store), names="value")
    run_btn.on_click(lambda btn, store=store: on_run(btn, store=store))
    reset_btn.on_click(lambda _: reset_weight_state(store))

    rank_box = _build_rank_options(store)
    manual_box = _build_manual_override(store)
    weight_box = _build_weighting_options(store)

    def _toggle_boxes(change: dict[str, Any], *, store: ParamStore) -> None:
        mode_val = change["new"] if isinstance(change, dict) else mode.value
        rank_box.layout.display = "flex" if mode_val == "rank" else "none"
        manual_box.layout.display = "flex" if mode_val == "manual" else "none"

    mode.observe(lambda ch, store=store: _toggle_boxes(ch, store=store), names="value")
    _toggle_boxes({"new": mode.value}, store=store)

    def _refresh_config_widgets() -> None:
        """Reflect a replacement config in the controls that remain mounted."""

        portfolio_cfg = _as_dict(store.cfg.get("portfolio"))
        mode.value = portfolio_cfg.get("selection_mode", "all")
        vol_adj.value = _as_dict(store.cfg.get("vol_adjust")).get("enabled", True)
        refreshed_export = _as_dict(store.cfg.get("export"))
        refreshed_formats = _normalized_export_formats(refreshed_export.get("formats"))
        fmt_dd.value = str(refreshed_formats[0]) if refreshed_formats else "xlsx"

        rank_box.children = _build_rank_options(store).children
        manual_box.children = _build_manual_override(store).children
        weight_box.children = _build_weighting_options(store).children
        _toggle_boxes({"new": mode.value}, store=store)

    step0 = _build_step0(store, on_config_replaced=_refresh_config_widgets)

    container = widgets.VBox(
        [
            step0,
            mode,
            vol_adj,
            rank_box,
            manual_box,
            weight_box,
            fmt_dd,
            theme,
            reset_btn,
            run_btn,
        ]
    )
    return container


__all__ = [
    "launch",
    "build_config_dict",
    "build_config_from_store",
    "load_state",
    "save_state",
    "reset_weight_state",
]
