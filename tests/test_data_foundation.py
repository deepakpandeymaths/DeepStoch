from pathlib import Path

import pytest

from deepstoch.data.loader import DataLoadError, load_csv
from deepstoch.data.split import chronological_split
from deepstoch.data.validation import validate_rows

DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_demo_dataset_loads_and_validates() -> None:
    headers, rows = load_csv(DEMO)
    result = validate_rows(headers, rows, time_column="timestamp", target_column="target")
    assert result.valid
    assert len(rows) == 48


def test_chronological_split_preserves_order() -> None:
    split = chronological_split(list(range(10)), train_fraction=0.6, validation_fraction=0.2)
    assert split.train == (0, 1, 2, 3, 4, 5)
    assert split.validation == (6, 7)
    assert split.test == (8, 9)


def test_invalid_target_is_reported() -> None:
    result = validate_rows(["timestamp", "target"], [{"timestamp": "2026-01-01", "target": "bad"}], time_column="timestamp", target_column="target")
    assert not result.valid
    assert any(issue.code == "non-numeric-target" for issue in result.errors)


def test_loader_rejects_non_csv() -> None:
    with pytest.raises(DataLoadError, match="CSV file does not exist"):
        load_csv(DEMO.with_suffix(".txt"))

