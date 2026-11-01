from deepstoch.evaluation.metrics import mae, mape, rmse
from deepstoch.experiments.baselines import evaluate_baselines
from deepstoch.models.baselines import moving_average_forecast, seasonal_naive_forecast


def test_baselines_return_requested_horizon() -> None:
    assert moving_average_forecast([1.0, 2.0, 3.0], 2, window=2) == [2.5, 2.5]
    assert seasonal_naive_forecast([1.0, 2.0, 3.0], 4, season_length=2) == [2.0, 3.0, 2.0, 3.0]


def test_metrics_are_deterministic() -> None:
    actual = [2.0, 4.0]
    predicted = [1.0, 5.0]
    assert mae(actual, predicted) == 1.0
    assert rmse(actual, predicted) == 1.0
    assert mape(actual, predicted) == 37.5


def test_baseline_evaluation_is_ranked_by_rmse() -> None:
    scores = evaluate_baselines([1.0, 2.0, 3.0, 4.0], [4.0, 5.0], 2)
    assert len(scores) == 4
    assert scores[0].rmse <= scores[-1].rmse
