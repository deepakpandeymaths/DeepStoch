from pathlib import Path
import importlib.util

import pytest

from deepstoch.data.loader import load_csv
from deepstoch.evaluation.calibration import summarize_calibration
from deepstoch.experiments.rolling import evaluate_repeated_rolling_neural_state, evaluate_rolling_ml_state, evaluate_rolling_neural_state


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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is optional")
def test_repeated_rolling_neural_evaluation_reports_seed_variation() -> None:
    rows = [
        {"timestamp": f"2025-01-01T{hour:02d}:00:00", "target": str(10 + hour), "temperature": str(20 + hour / 10)}
        for hour in range(12)
    ]
    result = evaluate_repeated_rolling_neural_state(rows, seeds=[1, 2], sequence_length=2, minimum_train=8, test_size=2, step=2, epochs=1, patience=1)
    assert result.seeds == [1, 2]
    assert result.folds_per_seed == 2
    assert len(result.seed_summaries) == 2
    assert result.std_rmse >= 0
    assert result.calibration["absolute_coverage_error"] >= 0


def test_calibration_summary_reports_signed_and_absolute_error() -> None:
    result = summarize_calibration([
        {"nominal_coverage": 0.9, "empirical_coverage": 0.8, "mean_interval_width": 10.0},
        {"nominal_coverage": 0.9, "empirical_coverage": 1.0, "mean_interval_width": 20.0},
    ])
    assert result.coverage_error == 0
    assert result.absolute_coverage_error == 0
    assert result.mean_interval_width == 15
    assert result.minimum_fold_coverage == 0.8
