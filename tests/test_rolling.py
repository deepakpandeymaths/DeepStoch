from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.experiments.rolling import evaluate_rolling_ml_state


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_rolling_evaluation_creates_multiple_temporal_folds() -> None:
    _, rows = load_csv(DEMO)
    result = evaluate_rolling_ml_state(rows, minimum_train=24, test_size=4, step=4)
    assert result.model == "linear_ml_plus_residual_state"
    assert result.folds == 6
    assert result.mean_rmse >= 0
    assert result.std_rmse >= 0
    assert len(result.fold_scores) == result.folds
