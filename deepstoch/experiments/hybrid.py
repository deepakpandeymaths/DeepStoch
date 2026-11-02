"""Phase 8 interpretable hybrid state-space experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from deepstoch.evaluation.metrics import mae, rmse
from deepstoch.stochastic.state_space import LocalLevelModel


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


def _variance_of_differences(values: list[float]) -> float:
    differences = [current - previous for previous, current in zip(values, values[1:])]
    if len(differences) < 2:
        return 1.0
    mean = sum(differences) / len(differences)
    return sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
