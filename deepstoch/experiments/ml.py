"""Phase 3 classical ML experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from deepstoch.data.features import build_features
from deepstoch.evaluation.metrics import mae, mape, rmse
from deepstoch.models.linear import LinearRegression
from deepstoch.models.tree import RandomForestRegressor


@dataclass(frozen=True)
class MLScore:
    model: str
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    mape: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_linear_regression(rows: list[dict[str, str]], target_column: str = "target", time_column: str = "timestamp", test_size: int = 8) -> MLScore:
    feature_rows = build_features(rows, target_column, time_column)
    if test_size < 1 or test_size >= len(feature_rows):
        raise ValueError("test-size must be at least 1 and smaller than the usable feature rows")
    train = feature_rows[:-test_size]
    test = feature_rows[-test_size:]
    model = LinearRegression().fit([item.features for item in train], [item.target for item in train])
    actual = [item.target for item in test]
    predicted = model.predict([item.features for item in test])
    return MLScore("linear_regression", len(train), len(test), mae(actual, predicted), rmse(actual, predicted), mape(actual, predicted))


def evaluate_random_forest(rows: list[dict[str, str]], target_column: str = "target", time_column: str = "timestamp", test_size: int = 8) -> MLScore:
    """Fit a reproducible random forest on the same leakage-safe features."""
    feature_rows = build_features(rows, target_column, time_column)
    if test_size < 1 or test_size >= len(feature_rows):
        raise ValueError("test-size must be at least 1 and smaller than the usable feature rows")
    train = feature_rows[:-test_size]
    test = feature_rows[-test_size:]
    model = RandomForestRegressor().fit([item.features for item in train], [item.target for item in train])
    actual = [item.target for item in test]
    predicted = model.predict([item.features for item in test])
    return MLScore("random_forest", len(train), len(test), mae(actual, predicted), rmse(actual, predicted), mape(actual, predicted))
