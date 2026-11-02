"""Phase 8 interpretable hybrid state-space experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from deepstoch.evaluation.metrics import mae, rmse
from deepstoch.data.features import build_features
from deepstoch.models.linear import LinearRegression
from deepstoch.stochastic.state_space import LocalLevelModel, LocalLinearTrendModel


@dataclass(frozen=True)
class HybridScore:
    model: str
    train_size: int
    test_size: int
    mae: float
    rmse: float
    interval_coverage: float
    mean_interval_width: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_local_level(rows: list[dict[str, str]], target_column: str = "target", test_size: int = 8) -> HybridScore:
    """Fit a latent-level process on the past and evaluate future observations."""
    values = [float(row[target_column]) for row in rows]
    if test_size < 1 or test_size >= len(values):
        raise ValueError("test-size must be at least 1 and smaller than the dataset")
    train, test = values[:-test_size], values[-test_size:]
    model = LocalLevelModel(process_variance=max(1.0, _variance_of_differences(train)), observation_variance=max(1.0, _variance_of_differences(train) / 2))
    forecast = model.fit(train).forecast(test_size)
    covered = sum(lower <= actual <= upper for actual, lower, upper in zip(test, forecast.lower, forecast.upper))
    return HybridScore("local_level_state_space", len(train), len(test), mae(test, forecast.mean), rmse(test, forecast.mean), covered / len(test), sum(upper - lower for lower, upper in zip(forecast.lower, forecast.upper)) / len(test))


def evaluate_local_trend(rows: list[dict[str, str]], target_column: str = "target", test_size: int = 8) -> HybridScore:
    """Fit a latent level-and-trend process on the past."""
    values = [float(row[target_column]) for row in rows]
    if test_size < 1 or test_size >= len(values):
        raise ValueError("test-size must be at least 1 and smaller than the dataset")
    train, test = values[:-test_size], values[-test_size:]
    variance = _variance_of_differences(train)
    model = LocalLinearTrendModel(level_variance=max(1.0, variance), trend_variance=max(0.1, variance / 10), observation_variance=max(1.0, variance / 2))
    forecast = model.fit(train).forecast(test_size)
    covered = sum(lower <= actual <= upper for actual, lower, upper in zip(test, forecast.lower, forecast.upper))
    return HybridScore("local_linear_trend_state_space", len(train), len(test), mae(test, forecast.mean), rmse(test, forecast.mean), covered / len(test), sum(upper - lower for lower, upper in zip(forecast.lower, forecast.upper)) / len(test))


def evaluate_ml_state_hybrid(rows: list[dict[str, str]], target_column: str = "target", time_column: str = "timestamp", test_size: int = 8) -> HybridScore:
    """Correct a linear ML forecast with a filtered latent residual state."""
    feature_rows = build_features(rows, target_column, time_column)
    if test_size < 1 or test_size >= len(feature_rows):
        raise ValueError("test-size must be at least 1 and smaller than the usable feature rows")
    train = feature_rows[:-test_size]
    test = feature_rows[-test_size:]
    model = LinearRegression().fit([item.features for item in train], [item.target for item in train])
    train_prediction = model.predict([item.features for item in train])
    residuals = [item.target - prediction for item, prediction in zip(train, train_prediction)]
    residual_variance = max(1.0, _variance_of_differences(residuals))
    residual_state = LocalLevelModel(process_variance=residual_variance, observation_variance=max(1.0, residual_variance / 2)).fit(residuals).forecast(test_size)
    base_prediction = model.predict([item.features for item in test])
    predictions = [base + correction for base, correction in zip(base_prediction, residual_state.mean)]
    actual = [item.target for item in test]
    lower = [prediction + correction for prediction, correction in zip(base_prediction, residual_state.lower)]
    upper = [prediction + correction for prediction, correction in zip(base_prediction, residual_state.upper)]
    covered = sum(low <= value <= high for value, low, high in zip(actual, lower, upper))
    return HybridScore("linear_ml_plus_residual_state", len(train), len(test), mae(actual, predictions), rmse(actual, predictions), covered / len(actual), sum(high - low for low, high in zip(lower, upper)) / len(actual))


def _variance_of_differences(values: list[float]) -> float:
    differences = [current - previous for previous, current in zip(values, values[1:])]
    if len(differences) < 2:
        return 1.0
    mean = sum(differences) / len(differences)
    return sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
