# Command Line Interface

The project ships with two convenience entry points once the package is
installed (or in editable mode via `pip install -e .`). Both commands share
the same underlying configuration pipeline that powers the Streamlit
experience.

## Launching the Streamlit application (`trend-app`)

```
trend-app
```

The command is a thin wrapper around `streamlit run streamlit_app/app.py`.
Any additional arguments are forwarded to Streamlit so you can tweak options
such as `--server.port` or `--server.headless` if required.

## Generating reports headlessly (`trend-run`)

The `trend-run` command executes the full backtest using a configuration file
and writes a self-contained HTML report. It accepts both YAML and TOML
configuration formats and automatically resolves relative paths the same way
as the Streamlit UI.

```
trend-run -c config/trend.yml -o reports/run.html
```

To override the returns CSV declared in the configuration, provide the
`--returns` option. When you want CSV/JSON/XLSX/TXT artifacts alongside the
HTML report, pass an output directory via `--artifacts` and optionally narrow
the formats with `--formats`:

