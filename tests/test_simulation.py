from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.experiments.simulation import simulate_dataset
from deepstoch.stochastic.bootstrap import simulate_bootstrap_paths, summarize_paths


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_bootstrap_simulation_is_reproducible() -> None:
    first = simulate_bootstrap_paths([10.0, 11.0], [1.0, -1.0], 3, paths=4, seed=7)
    second = simulate_bootstrap_paths([10.0, 11.0], [1.0, -1.0], 3, paths=4, seed=7)
    assert first == second
    assert len(first) == 4 and len(first[0]) == 3


def test_path_summary_reports_threshold_probability() -> None:
    result = summarize_paths([[1.0, 5.0], [2.0, 3.0]], threshold=4.0)
    assert result.quantiles["0.5"] == [1.5, 4.0]
    assert result.probability_exceeds_threshold == 0.5


def test_dataset_simulation_returns_expected_shape() -> None:
    _, rows = load_csv(DEMO)
    result = simulate_dataset(rows, horizon=6, paths=25, seed=3, threshold=700)
    assert result.paths == 25
    assert result.horizon == 6
    assert len(result.quantiles["0.5"]) == 6
    assert 0 <= result.probability_exceeds_threshold <= 1
