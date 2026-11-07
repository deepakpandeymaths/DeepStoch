from pathlib import Path
import importlib.util

import pytest

from deepstoch.data.loader import load_csv
from deepstoch.experiments.rolling import evaluate_rolling_ml_state, evaluate_rolling_neural_state


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_rolling_evaluation_creates_multiple_temporal_folds() -> None:
    _, rows = load_csv(DEMO)
    result = evaluate_rolling_ml_state(rows, minimum_train=24, test_size=4, step=4)
    assert result.model == "linear_ml_plus_residual_state"
    assert result.folds == 6
    assert result.mean_rmse >= 0
    assert result.std_rmse >= 0
    assert len(result.fold_scores) == result.folds


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is optional")
def test_rolling_neural_evaluation_retrains_per_temporal_fold() -> None:
    rows = [
        {"timestamp": f"2025-01-01T{hour:02d}:00:00", "target": str(10 + hour), "temperature": str(20 + hour / 10)}
        for hour in range(16)
    ]
    result = evaluate_rolling_neural_state(rows, sequence_length=2, minimum_train=8, test_size=2, step=2, epochs=2, patience=2)
    assert result.model == "neural_state_space"
    assert result.folds == 4
    assert len(result.fold_scores) == result.folds
    assert 0 <= result.mean_coverage <= 1
