"""CSV loading with explicit, predictable parsing rules."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeAlias

Row: TypeAlias = dict[str, str]


class DataLoadError(ValueError):
    """Raised when a CSV file cannot be loaded safely."""


def load_csv(path: Path, *, encoding: str = "utf-8-sig") -> tuple[list[str], list[Row]]:
    """Load a CSV file and return its header and rows.

    The loader deliberately preserves values as strings. Type conversion belongs to
    validation/preprocessing so that the original input is not silently changed.
    """

    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"CSV file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise DataLoadError(f"Expected a .csv file, received: {path.name}")

    try:
        with path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataLoadError("CSV file has no header row")
            headers = [name.strip() for name in reader.fieldnames]
            if any(not name for name in headers):
                raise DataLoadError("CSV header contains an empty column name")
            if len(set(headers)) != len(headers):
                raise DataLoadError("CSV header contains duplicate column names")

            rows: list[Row] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise DataLoadError(f"Row {line_number} has more values than the header")
                rows.append({key.strip(): (value or "").strip() for key, value in row.items()})
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"CSV is not valid UTF-8: {path.name}") from exc
    except csv.Error as exc:
        raise DataLoadError(f"CSV parsing failed at {path.name}: {exc}") from exc

    return headers, rows

