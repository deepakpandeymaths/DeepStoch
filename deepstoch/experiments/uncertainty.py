"""Phase 5 uncertainty experiment using a calibrated linear forecast."""

from __future__ import annotations

from deepstoch.data.features import build_features
from deepstoch.models.linear import LinearRegression
from deepstoch.uncertainty.conformal import ConformalInterval, split_conformal_interval


def evaluate_conformal_linear(
    rows: list[dict[str, str]],
    target_column: str = "target",
    time_column: str = "timestamp",
    calibration_size: int = 8,
    test_size: int = 8,
    nominal_coverage: float = 0.9,
) -> ConformalInterval:
    """Fit on the earliest segment and calibrate/test strictly in time order."""
    feature_rows = build_features(rows, target_column, time_column)
    if calibration_size < 1 or test_size < 1 or calibration_size + test_size >= len(feature_rows):
        raise ValueError("calibration-size and test-size must leave at least one training row")
    train_end = len(feature_rows) - calibration_size - test_size
    train = feature_rows[:train_end]
    calibration = feature_rows[train_end:train_end + calibration_size]
    test = feature_rows[train_end + calibration_size:]
    model = LinearRegression().fit([item.features for item in train], [item.target for item in train])
    calibration_actual = [item.target for item in calibration]
    test_actual = [item.target for item in test]
    calibration_predicted = model.predict([item.features for item in calibration])
    test_predicted = model.predict([item.features for item in test])
    return split_conformal_interval(calibration_actual, calibration_predicted, test_actual, test_predicted, nominal_coverage=nominal_coverage)


def evaluate_conformal_levels(
    rows: list[dict[str, str]], levels: list[float], target_column: str = "target", time_column: str = "timestamp", calibration_size: int = 8, test_size: int = 8
) -> list[ConformalInterval]:
    """Evaluate several nominal coverage levels with one chronological fit."""
    if not levels:
        raise ValueError("at least one coverage level is required")
    feature_rows = build_features(rows, target_column, time_column)
    if calibration_size < 1 or test_size < 1 or calibration_size + test_size >= len(feature_rows):
        raise ValueError("calibration-size and test-size must leave at least one training row")
    train_end = len(feature_rows) - calibration_size - test_size
    train = feature_rows[:train_end]
    calibration = feature_rows[train_end:train_end + calibration_size]
    test = feature_rows[train_end + calibration_size:]
    model = LinearRegression().fit([item.features for item in train], [item.target for item in train])
    calibration_actual = [item.target for item in calibration]
    test_actual = [item.target for item in test]
    calibration_predicted = model.predict([item.features for item in calibration])
    test_predicted = model.predict([item.features for item in test])
    return [split_conformal_interval(calibration_actual, calibration_predicted, test_actual, test_predicted, nominal_coverage=level) for level in levels]
