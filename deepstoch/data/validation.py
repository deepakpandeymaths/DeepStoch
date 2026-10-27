"""Dataset validation rules for the first DeepStoch data foundation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from deepstoch.data.loader import Row


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable validation finding."""

    code: str
    message: str
    row: int | None = None
    column: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome with errors, warnings, and parsed timestamps."""

    valid: bool
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    timestamps: tuple[datetime | None, ...]


def _parse_timestamp(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def validate_rows(
    headers: Iterable[str],
    rows: list[Row],
    *,
    time_column: str,
    target_column: str,
) -> ValidationResult:
    """Validate required columns, timestamps, target values, duplicates, and ordering."""

    headers = tuple(headers)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    for required in (time_column, target_column):
        if required not in headers:
            errors.append(ValidationIssue("missing-column", f"Required column is missing: {required}", column=required))

    timestamps: list[datetime | None] = []
    if time_column in headers:
        for index, row in enumerate(rows, start=2):
            raw = row.get(time_column, "")
            parsed = _parse_timestamp(raw)
            timestamps.append(parsed)
            if not raw:
                errors.append(ValidationIssue("missing-timestamp", "Timestamp is empty", index, time_column))
            elif parsed is None:
                errors.append(ValidationIssue("invalid-timestamp", f"Cannot parse timestamp: {raw}", index, time_column))
    else:
        timestamps = [None] * len(rows)

    if target_column in headers:
        for index, row in enumerate(rows, start=2):
            value = row.get(target_column, "").strip()
            if not value:
                errors.append(ValidationIssue("missing-target", "Target value is empty", index, target_column))
            else:
                try:
                    float(value)
                except ValueError:
                    errors.append(ValidationIssue("non-numeric-target", f"Target is not numeric: {value}", index, target_column))

    seen: dict[datetime, int] = {}
    for index, timestamp in enumerate(timestamps, start=2):
        if timestamp is None:
            continue
        if timestamp in seen:
            errors.append(ValidationIssue("duplicate-timestamp", f"Duplicate timestamp; first seen at row {seen[timestamp]}", index, time_column))
        else:
            seen[timestamp] = index

    valid_timestamps = [value for value in timestamps if value is not None]
    if valid_timestamps and valid_timestamps != sorted(valid_timestamps):
        warnings.append(ValidationIssue("unsorted-timestamps", "Timestamps are not in chronological order", column=time_column))

    if not rows:
        warnings.append(ValidationIssue("empty-dataset", "Dataset contains a header but no rows"))

    return ValidationResult(not errors, tuple(errors), tuple(warnings), tuple(timestamps))

