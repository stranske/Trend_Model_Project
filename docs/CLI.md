# Trend Model CLI Quickstart

This guide walks through the two console entry points exposed by the Trend Model
package after it has been installed (for example via `pip install -e .`).

## Prerequisites

1. Create and activate a virtual environment (optional but recommended).
2. Install the project in editable mode so the console scripts are available:

   ```bash
   pip install -e .[app]
   ```

   The optional `app` extra pulls in Streamlit so that the `trend-app` command
   can launch the interactive UI.

3. Generate the demo dataset that the sample configuration relies on:

   ```bash
   python scripts/generate_demo.py
   ```

   This writes `demo/demo_returns.csv`, which the sample configuration file
   references.

## Launching the Streamlit UI (`trend-app`)

Run the `trend-app` console script to launch the Streamlit interface bundled in
`streamlit_app/app.py`:

```bash
trend-app
```

The command proxies directly to `streamlit run streamlit_app/app.py`, so any
arguments you provide are forwarded to Streamlit itself. For example, to launch
headless on a specific port:

```bash
trend-app --server.headless true --server.port 8502
```

## Running analyses headlessly (`trend-run`)

The `trend-run` console script executes the full volatility-adjusted trend
pipeline using a YAML or TOML configuration file and produces an HTML report by
default. The repository now ships with a TOML example at `config/trend.toml`
that mirrors the demonstration YAML configuration.

Generate the demo dataset first (see the prerequisites above), then invoke the
command:

```bash
trend-run -c config/trend.toml -o reports/cli_demo.html
```

The example configuration writes the report to the location provided via
`-o/--output`. You can also direct the command to export CSV, JSON, XLSX, or TXT
artefacts by pointing `--artefacts` at a directory and optionally specifying the
formats to emit.

Example:

```bash
trend-run -c config/trend.toml \
  -o reports/cli_demo.html \
  --artefacts reports/artefacts \
  --formats csv json xlsx
```

### PDF export

Pass `--pdf` to render a PDF alongside the HTML report. This requires the
`fpdf2` dependency (install with `pip install "fpdf2>=2.7"`). When enabled, the
command writes `<output>.pdf` next to the HTML file.

### Configuration tips

* Relative paths inside the configuration are resolved relative to the config
  file, so `config/trend.toml` can reference `demo/demo_returns.csv` without an
  absolute path.
* TOML and YAML configs share the same schema. You can base your own TOML files
  on the provided example or convert existing YAML configs by matching the key
  structure.
* The `seed` parameter ensures deterministic behaviour. Adjust it or pass
  `--seed` on the command line to override per run.

---

For advanced usage—including multi-period backtesting and custom selector or
weighting logic—refer to the broader documentation under `docs/` and the
configuration schema in `config/defaults.yml`.
