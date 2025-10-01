# Market Data Validation Contract

## Schema library decision

We selected **Pandera** for DataFrame validation because it natively understands
pandas semantics (column-wise coercion, nullable floats, datetime indices) and
lets us express checks at the frame level. Pydantic is excellent for record
validation, but would have required manual loops to normalise timestamps and
coerce each column. Pandera allowed us to centralise those rules in a single
schema, while still raising typed exceptions when constraints fail.

## `validate_market_data` contract

```python
from trend_analysis.io.market_data import validate_market_data

clean = validate_market_data(raw_frame)
```

* Returns a copy of the input with a **DatetimeIndex** named `Date`, sorted,
  unique, and timezone-naive.
* All value columns are coerced to floating point. Pandera raises an error when
  a column cannot be converted or contains only missing values.
* The validator infers cadence (`daily`, `monthly`, etc.) and whether the frame
  represents **returns** or **prices**. This metadata is stored in
  `clean.attrs["market_data"]` alongside ISO formatted `start` / `end`
  timestamps. Convenience keys `clean.attrs["market_data_mode"]` and
  `clean.attrs["market_data_frequency"]` mirror those values for quick access.
* Failures raise `MarketDataValidationError` whose message is safe to surface in
  user interfaces (Streamlit upload banner, CLI stderr). The CLI now exits with
  the same message, and the Streamlit page displays it in a single error banner.

## Common failure modes

| Failure | Message | Resolution |
| --- | --- | --- |
| Duplicate timestamps | `Duplicate timestamps detected: …` | De-duplicate upstream (e.g. group by date) before ingest. |
| Unsorted dates | `Timestamps must be sorted in ascending order.` | Sort chronologically; the validator never reorders rows. |
| Mixed cadence | `Mixed sampling cadence detected …` | Resample to a single cadence (e.g. all month-end). |
| Mode ambiguity | `Unable to determine if the dataset is in returns or price mode.` | Confirm values are either returns (≈±100%) or raw prices. |
| Missing symbols | `No data columns provided alongside Date column.` | Include at least one numeric column besides `Date`. |

## Downstream usage

* **CLI** (`trend-model run`) calls `load_csv(..., errors="raise")` so schema
  failures stop before modelling starts and print the validator message.
* **Streamlit upload** (`1_Upload` page) surfaces the same message in a single
  error banner and keeps the upload state unchanged when validation fails.
* **Reusable helpers** (`load_parquet`, `validate_dataframe`,
  `load_and_validate_upload`) all delegate to `validate_market_data` so CSV,
  Parquet, and in-memory DataFrames are normalised consistently.
