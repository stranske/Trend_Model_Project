# Browser demo (stlite / Pyodide) — issue #5343

A zero-install, browser-reachable demo of the Portfolio Simulator that runs the
**real** Streamlit app (`streamlit_app/app.py`) and the deterministic
`trend_analysis` engine entirely in the browser via
[stlite](https://github.com/whitphx/stlite) + Pyodide.

This is the WASM/Pyodide/stlite demo required by the owner decision in
[issue #5343](https://github.com/stranske/Trend_Model_Project/issues/5343). It is
deliberately **not** a Streamlit Community Cloud deployment.

## Runtime profiles

The active profile is selected with the `?profile=` query parameter and enforced
by [`streamlit_app/demo_profile.py`](../../streamlit_app/demo_profile.py).

| Profile | Default | LLM / LangChain UI | Custom analysis | Data upload | Data source |
| --- | --- | --- | --- | --- | --- |
| `presentation_safe` | ✅ | hidden | hidden | hidden | bundled synthetic (`demo/demo_returns.csv`) |
| `public_llm_demo` | | enabled | enabled | enabled | bundled synthetic by default |

- **`presentation_safe`** is the default so a locked-down work PC or a live
  presentation never surfaces an LLM or upload control unexpectedly. The
  deterministic synthetic-data flow still runs end to end. Its requirement set
  omits LangChain, so after the page loads there is no LLM dependency footprint
  and no unexpected egress.
- **`public_llm_demo`** adds the LangChain stack (pinned to match
  `pyproject.toml`) and exposes the LLM UI. **No secrets are bundled**; any
  provider key/endpoint must be entered at runtime.

A reviewer can switch modes live from the **Demo mode** selector in the sidebar
(or by changing `?profile=`) and watch the LLM / custom-analysis / upload
surfaces appear and disappear.

## Build

The page is driven by a generated `manifest.json` (entrypoint, the source-file
list, and the per-profile requirement sets). Regenerate it whenever the app or
engine source changes:

```bash
python scripts/build_wasm_demo.py          # writes demo/wasm/manifest.json
python scripts/build_wasm_demo.py --check   # CI guard: fail if stale
```

`index.html` fetches `manifest.json`, then fetches each listed source file from
the publish base (default `./app/`, override with `?base=`), reconstructs the
in-browser filesystem, installs the profile's requirements under Pyodide, and
mounts the app.

The runtime itself is vendored under `demo/wasm/vendor/`: stlite
`@0.79.4`, Pyodide `0.27.2`, and the Pyodide wheels required by the default
`presentation_safe` profile. The originally referenced stlite `0.79.3` package
is not published on npm/jsDelivr, so the local bundle uses the nearest
available patch release and `build_wasm_demo.py --check` verifies the required
runtime files are present.

`plotly` (used by the **Monte Carlo** page's `st.plotly_chart` surfaces) is
pure-PyPI and not in the Pyodide lock, so its wheel is vendored separately under
`demo/wasm/vendor/pypi/` and installed by passing an absolute same-origin wheel
URL to micropip — `index.html` rewrites every `*.whl` requirement to
`new URL(path, location.href)` because micropip refuses a bare-relative wheel
URL (it treats it as `file://`). plotly 6.x's `plotly.express` requires
`narwhals>=1.15.1`, newer than the lock's 1.10.0, so the lock is bumped in place
to narwhals 1.15.1 (a single narwhals also satisfies the lock's altair).
Plotly.js renders inside the bundled stlite frontend — no `cdn.plot.ly` fetch —
so it does not need separate vendoring. Re-fetch all of the above with
`python scripts/fetch_offline_runtime.py`.

## Deploy

The public GitHub Pages target is
<https://stranske.github.io/Trend_Model_Project/>. The default profile is
presentation-safe; the explicit LLM profile is available at
<https://stranske.github.io/Trend_Model_Project/?profile=public_llm_demo>.

Pushes to `phase-3` that change the browser-demo inputs run
`.github/workflows/pages-wasm-demo.yml`. The workflow builds a self-contained
artifact with:

```bash
python scripts/build_wasm_site.py --output dist/wasm-demo
```

and publishes it with GitHub Pages. The manual layout below remains valid for
another static host.

The demo is a static bundle plus the published application source subset:

1. `python scripts/build_wasm_demo.py`
2. Publish `demo/wasm/index.html` and `demo/wasm/manifest.json` at the site root.
3. Publish this repository's `streamlit_app/`, `src/trend_analysis/`, and `demo/`
   subset under the base directory (default `./app/`) so the paths in
   `manifest.json` resolve.
4. Open `index.html` — defaults to `presentation_safe`; append
   `?profile=public_llm_demo` for the LLM mode.

> **Public LLM profile compatibility spike (follow-up).** The default
> `presentation_safe` profile uses only vendored Pyodide packages. If a
> provider-specific LangChain wheel is not browser-compatible for
> `public_llm_demo`, keep the stlite app and isolate the provider call behind a
> runtime-configured endpoint/adapter (per the issue's implementation notes)
> rather than removing LangChain from the public profile.

## Verification checklist

Capture this evidence on the deployed URL (PR acceptance criteria for #5343):

- [ ] Live URL loads `index.html` in a browser with no local install.
- [ ] `presentation_safe`: deterministic synthetic-data demo runs end to end;
      no LLM, custom-analysis, or upload controls are visible.
- [ ] `presentation_safe`: browser network panel shows no `cdn.jsdelivr.net`
      requests and no unexpected egress after the local static bundle loads.
- [ ] `public_llm_demo`: the LangChain LLM UI is visible; switching modes in the
      sidebar changes the visible surfaces.
- [ ] `public_llm_demo`: network panel shows only the expected configured
      provider traffic when the LLM feature is exercised, and no committed
      secrets.
- [ ] Screenshots attached for both modes.
