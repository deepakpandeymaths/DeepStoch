from pathlib import Path

from deepstoch.data.features import FEATURE_NAMES, build_features
from deepstoch.data.loader import load_csv
from deepstoch.experiments.ml import evaluate_linear_regression


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_features_use_prior_targets_only() -> None:
    _, rows = load_csv(DEMO)
    features = build_features(rows)
    assert len(FEATURE_NAMES) == 8
    assert features[0].index == 3
    assert features[0].features[0] == float(rows[2]["target"])
    assert features[0].target == float(rows[3]["target"])


def test_linear_regression_evaluation_returns_finite_metrics() -> None:
    _, rows = load_csv(DEMO)
    score = evaluate_linear_regression(rows, test_size=8)
    assert score.model == "linear_regression"
    assert score.train_rows == 37
    assert score.test_rows == 8
    assert score.rmse >= 0
