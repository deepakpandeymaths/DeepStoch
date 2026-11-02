from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.experiments.uncertainty import evaluate_conformal_levels, evaluate_conformal_linear
from deepstoch.uncertainty.conformal import split_conformal_interval


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_conformal_radius_is_calibrated_from_residuals() -> None:
    result = split_conformal_interval([1.0, 2.0, 4.0], [0.0, 3.0, 3.0], [2.0, 10.0], [2.0, 7.0], nominal_coverage=0.9)
    assert result.radius == 1.0
    assert result.empirical_coverage == 0.5
    assert result.mean_width == 2.0


def test_uncertainty_experiment_returns_calibrated_report() -> None:
    _, rows = load_csv(DEMO)
    result = evaluate_conformal_linear(rows, calibration_size=8, test_size=8)
    assert result.calibration_size == 8
    assert result.test_size == 8
    assert 0 <= result.empirical_coverage <= 1
    assert result.radius >= 0


def test_multiple_coverage_levels_are_ordered_by_width() -> None:
    _, rows = load_csv(DEMO)
    results = evaluate_conformal_levels(rows, [0.8, 0.9, 0.95], calibration_size=8, test_size=8)
    assert [result.nominal_coverage for result in results] == [0.8, 0.9, 0.95]
    assert results[0].mean_width <= results[1].mean_width <= results[2].mean_width
