"""Walk-forward evaluation of Phase 2 baseline forecasters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from deepstoch.evaluation.metrics import mae, mape, rmse
from deepstoch.models.baselines import (
    last_value_forecast,
    mean_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
)

ForecastFunction = Callable[[list[float], int], list[float]]


@dataclass(frozen=True)
class BaselineScore:
    model: str
    horizon: int
    mae: float
    rmse: float
    mape: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_baselines(history: list[float], actual: list[float], horizon: int) -> list[BaselineScore]:
    """Compare baselines using one fixed holdout horizon."""
    if horizon != len(actual):
        raise ValueError("horizon must equal the number of actual holdout values")
    models: list[tuple[str, ForecastFunction]] = [
        ("last_value", last_value_forecast),
        ("mean", mean_forecast),
        ("moving_average_3", lambda values, size: moving_average_forecast(values, size, 3)),
        ("seasonal_naive_24", lambda values, size: seasonal_naive_forecast(values, size, 24)),
    ]
    scores = []
    for name, forecaster in models:
        predicted = forecaster(history, horizon)
        scores.append(BaselineScore(name, horizon, mae(actual, predicted), rmse(actual, predicted), mape(actual, predicted)))
    return sorted(scores, key=lambda score: score.rmse)
