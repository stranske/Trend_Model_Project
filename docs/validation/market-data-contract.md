# Market data ingest contract

The ingest pipeline now enforces a single validation flow for CSV, Parquet, and
in-memory DataFrame uploads.  This ensures both the Streamlit app and the CLI
surface identical, actionable feedback when data does not meet expectations.

## Dependency choice

We elected to stay within the existing dependency stack and build on top of
**Pydantic v2** instead of introducing a new schema library such as Pandera.
Pydantic is already required throughout the project, which keeps the validator
lightweight while still giving us structured metadata objects for downstream
consumers.  The DataFrame-specific checks (index ordering, duplicates, cadence
consistency, and value-type inference) are implemented with idiomatic
`pandas`, and the resulting metadata is serialised through a Pydantic model to
guarantee type-safety for callers.

## Validation steps

`validate_market_data(data: DataFrame)` performs the following checks:

1. **Index normalisation** – accepts either a `Date` column or an existing
   `DatetimeIndex`, coercing to a timezone-naïve index named `Date`.
2. **Ordering & duplicates** – rejects unsorted indices and duplicate
   timestamps with explicit examples in the error message.
3. **Numeric coercion** – converts all value columns to numeric, reporting any
   column that ends up without usable data.
4. **Cadence inference** – validates that the index spacing is consistent,
   raising when the cadence cannot be inferred (e.g., irregular gaps).
5. **Mode detection** – classifies the dataset as price- or returns-based and
   prevents mixed representations.

On success the helper returns a `ValidatedMarketData` container pairing the
normalised frame with a `MarketDataMetadata` model.  The metadata captures mode,
frequency (label and canonical code), inferred symbols, row counts, and the
start/end timestamps.  The metadata is also attached to the returned DataFrame
via `df.attrs["market_data"]["metadata"]` for convenience.

### Metadata propagation

- `ValidatedMarketData.frame` always carries a timezone-naïve `DatetimeIndex`
  named `Date`.  The `df.attrs["market_data"]` dictionary exposes:
  - `metadata`: the raw `MarketDataMetadata` Pydantic model.
  - `mode` / `mode_enum`: returns vs prices (string + enum).
  - `frequency` / `frequency_code`: human label and pandas offset alias.
  - `symbols`, `columns`, `rows`, and the `start`/`end` ISO-8601 timestamps.
- The Streamlit helper (`trend_portfolio_app.data_schema.SchemaMeta`) mirrors
  those fields and stores them in `st.session_state["schema_meta"]` together
  with a lightweight validation report so downstream pages can surface mode and
  cadence hints.
- CLI entry points keep the attrs intact, enabling exporters and bundle
  builders to introspect the ingest metadata without recomputing it.

Validation failures raise `MarketDataValidationError` with a single, bullet
point formatted message alongside a populated `issues` list.  This message is
displayed verbatim in both the CLI and the Streamlit UI, while callers that
need structured details can read the `issues` attribute to satisfy the
acceptance criteria.

## Integration points

- `load_market_data_csv` / `load_market_data_parquet` power the CLI and the
  `trend_portfolio_app` upload helpers, returning validated frames and
  metadata.
- The Streamlit upload flow funnels everything through `streamlit_app.state` so
  a single error banner is shown when validation fails and successful uploads
  persist metadata/validation reports in `st.session_state` for later pages.
- CLI runs exit with status code `1` and echo the same message to `stderr` so
  scripted pipelines can fail fast.

Refer to the unit tests in `tests/test_validators.py`,
`tests/test_io_validators_additional.py`, and
`tests/app/test_upload_page.py` for examples covering returns vs price
validation, failure messaging, and the Streamlit banner contract.
