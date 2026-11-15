from pathlib import Path

import pytest

from deepstoch.api.service import dataset_summary, model_comparison


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
