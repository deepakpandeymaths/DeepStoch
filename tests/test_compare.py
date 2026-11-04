from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.experiments.compare import compare_models


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"
CHECKPOINT = Path(__file__).parents[1] / "artifacts" / "neural_state_space.pt"


def test_comparison_report_contains_all_available_model_families() -> None:
    _, rows = load_csv(DEMO)
    result = compare_models(rows, test_size=8, neural_checkpoint=CHECKPOINT if CHECKPOINT.exists() else None)
    names = {row.model for row in result}
    assert "linear_regression" in names
    assert "random_forest" in names
    assert "linear_ml_plus_residual_state" in names
    assert len(result) >= 7
