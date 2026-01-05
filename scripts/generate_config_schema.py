"""Generate configuration JSON schema artifacts."""

from __future__ import annotations

from trend_analysis.config.schema_generator import write_schema_files


def main() -> None:
    schema_path, compact_path = write_schema_files()
    print(f"Wrote {schema_path} and {compact_path}")


if __name__ == "__main__":
    main()
