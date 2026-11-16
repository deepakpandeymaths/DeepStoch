"""Dependency-light service functions used by the optional HTTP API."""

from __future__ import annotations

from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.data.split import chronological_split
from deepstoch.data.validation import validate_rows
from deepstoch.experiments.compare import compare_models
from deepstoch.experiments.uncertainty import evaluate_conformal_linear
from deepstoch.experiments.simulation import simulate_dataset


def dataset_summary(
    path: str | Path,
    *,
    time_column: str = "timestamp",
    target_column: str = "target",
) -> dict[str, object]:
    """Return a safe, read-only summary for a local CSV dataset."""
    dataset_path = Path(path)
    headers, rows = load_csv(dataset_path)
    validation = validate_rows(headers, rows, time_column=time_column, target_column=target_column)
    split = chronological_split(rows) if rows else None
    return {
        "path": str(dataset_path),
        "columns": headers,
        "rows": len(rows),
        "validation": {
            "valid": validation.valid,
            "errors": [issue.message for issue in validation.errors],
            "warnings": [issue.message for issue in validation.warnings],
        },
        "split": None if split is None else {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
    }


def model_comparison(
    path: str | Path,
    *,
    time_column: str = "timestamp",
    target_column: str = "target",
    test_size: int = 8,
) -> dict[str, object]:
    """Return the unified chronological comparison for a local CSV dataset."""
    headers, rows = load_csv(Path(path))
    validation = validate_rows(headers, rows, time_column=time_column, target_column=target_column)
    if not validation.valid:
        raise ValueError("dataset failed validation; call dataset_summary for details")
    return {
        "dataset": dataset_summary(path, time_column=time_column, target_column=target_column),
        "test_size": test_size,
        "models": [score.as_dict() for score in compare_models(rows, target_column=target_column, time_column=time_column, test_size=test_size)],
    }


def uncertainty_report(
    path: str | Path,
    *,
    time_column: str = "timestamp",
    target_column: str = "target",
    calibration_size: int = 8,
    test_size: int = 8,
    coverage: float = 0.9,
) -> dict[str, object]:
    """Return a chronological split-conformal interval report for a CSV dataset."""
    headers, rows = load_csv(Path(path))
    validation = validate_rows(headers, rows, time_column=time_column, target_column=target_column)
    if not validation.valid:
        raise ValueError("dataset failed validation; call dataset_summary for details")
    interval = evaluate_conformal_linear(rows, target_column, time_column, calibration_size, test_size, coverage)
    return {"dataset": dataset_summary(path, time_column=time_column, target_column=target_column), "interval": interval.as_dict()}


def simulation_report(
    path: str | Path,
    *,
    target_column: str = "target",
    horizon: int = 24,
    paths: int = 1000,
    seed: int = 42,
    threshold: float | None = None,
) -> dict[str, object]:
    """Return reproducible bootstrap future trajectories for a local CSV."""
    headers, rows = load_csv(Path(path))
    validation = validate_rows(headers, rows, time_column="timestamp", target_column=target_column)
    if not validation.valid:
        raise ValueError("dataset failed target validation")
    result = simulate_dataset(rows, target_column, horizon, paths, seed, threshold)
    return {"path": str(path), "simulation": result.as_dict()}
