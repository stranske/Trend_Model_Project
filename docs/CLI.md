# Trend Model Command Line Interface

The project ships two lightweight console entry points for launching the
Streamlit application and executing headless analysis runs. Install the package
in editable mode during development:

```bash
pip install -e .[app]
```

The optional `app` extra pulls in Streamlit and web dependencies required for
the GUI launcher.

## `trend-app`

Launch the Streamlit interface that mirrors the web application shipped with
the repository:

```bash
trend-app
```

Any arguments after `--` are forwarded directly to `streamlit run`. For
example, to choose a custom port:

```bash
trend-app -- --server.port 8888 --server.headless true
```

## `trend-run`

Run the analysis pipeline from a configuration file and emit an HTML summary
report. The command supports both YAML and TOML inputs:

```bash
trend-run -c config/demo.yml -o reports/demo.html
```

Pass a TOML configuration the same way:

```bash
trend-run -c config/trend.toml -o reports/trend.html
```

Key flags:

* `--returns PATH` – override the CSV defined in the configuration.
* `--out DIR` – write CSV/JSON/XLSX/TXT artefacts to `DIR`.
* `--formats csv json` – restrict export formats (requires `--out`).
* `--pdf` – generate a PDF alongside the HTML report.
* `--seed VALUE` – force a deterministic random seed.

The command prints a human readable summary to stdout and reports where the HTML
and optional PDF outputs were written.

