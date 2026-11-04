from pathlib import Path

import pytest

from deepstoch.data.loader import load_csv
from deepstoch.experiments.neural_eval import evaluate_neural_state_space


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"
CHECKPOINT = Path(__file__).parents[1] / "artifacts" / "neural_state_space.pt"


def test_neural_checkpoint_evaluation_returns_metrics() -> None:
    if not CHECKPOINT.exists():
        pytest.skip("neural checkpoint has not been trained")
    _, rows = load_csv(DEMO)
    result = evaluate_neural_state_space(rows, CHECKPOINT, sequence_length=6, test_size=8)
    assert result.model == "neural_state_space"
    assert result.test_size == 8
    assert result.mae >= 0
    assert 0 <= result.empirical_coverage <= 1
