from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.experiments.hybrid import evaluate_local_level
from deepstoch.stochastic.state_space import LocalLevelModel


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_local_level_forecast_has_growing_uncertainty() -> None:
    forecast = LocalLevelModel(process_variance=1, observation_variance=1).fit([10, 11, 10]).forecast(4)
    assert len(forecast.mean) == 4
    assert forecast.upper[0] - forecast.lower[0] < forecast.upper[-1] - forecast.lower[-1]


def test_state_space_experiment_returns_metrics() -> None:
    _, rows = load_csv(DEMO)
    result = evaluate_local_level(rows, test_size=8)
    assert result.model == "local_level_state_space"
    assert result.train_size == 40
    assert 0 <= result.interval_coverage <= 1
    assert result.mean_interval_width > 0
