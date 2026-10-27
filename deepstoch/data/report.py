"""Human- and machine-readable dataset reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from deepstoch.data.loader import Row
from deepstoch.data.validation import ValidationResult


@dataclass(frozen=True)
class DatasetReport:
    path: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    missing_values: dict[str, int]
    validation_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    train_count: int
    validation_count: int
    test_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_report(
    path: Path,
    headers: list[str],
    rows: list[Row],
    validation: ValidationResult,
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> DatasetReport:
    missing_values = {column: sum(not row.get(column, "").strip() for row in rows) for column in headers}
    return DatasetReport(
        path=str(Path(path)),
        row_count=len(rows),
        column_count=len(headers),
        columns=tuple(headers),
        missing_values=missing_values,
        validation_valid=validation.valid,
        errors=tuple(issue.message for issue in validation.errors),
        warnings=tuple(issue.message for issue in validation.warnings),
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
    )

