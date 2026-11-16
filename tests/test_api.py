from pathlib import Path

import pytest

from deepstoch.api.service import dataset_summary, model_comparison, simulation_report, uncertainty_report


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_dataset_summary_is_read_only_and_reports_chronological_split() -> None:
    result = dataset_summary(DEMO)
    assert result["rows"] == 48
    assert result["validation"]["valid"] is True
    assert result["split"] == {"train": 33, "validation": 7, "test": 8}


def test_model_comparison_service_returns_dashboard_ready_rows() -> None:
    result = model_comparison(DEMO, test_size=8)
    names = {row["model"] for row in result["models"]}
    assert result["test_size"] == 8
    assert "linear_regression" in names
    assert "linear_ml_plus_residual_state" in names


def test_uncertainty_service_returns_interval_diagnostics() -> None:
    result = uncertainty_report(DEMO, calibration_size=8, test_size=8, coverage=0.9)
    interval = result["interval"]
    assert interval["nominal_coverage"] == 0.9
    assert interval["test_size"] == 8
    assert interval["mean_width"] >= 0


def test_simulation_service_returns_reproducible_trajectory_summary() -> None:
    result = simulation_report(DEMO, horizon=4, paths=12, seed=7, threshold=700)
    simulation = result["simulation"]
    assert simulation["paths"] == 12
    assert simulation["horizon"] == 4
    assert len(simulation["quantiles"]["0.5"]) == 4
    assert 0 <= simulation["probability_exceeds_threshold"] <= 1


def test_fastapi_adapter_has_explicit_optional_dependency_boundary() -> None:
    try:
        from deepstoch.api.app import create_app
        import importlib.util
        if importlib.util.find_spec("fastapi") is None:
            with pytest.raises(RuntimeError, match="pip install.*api"):
                create_app()
        else:
            assert create_app().title == "DeepStoch Local Experiment API"
    except ImportError:
        pytest.skip("optional API adapter cannot be imported")
