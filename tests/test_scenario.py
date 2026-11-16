import pytest

from deepstoch.scenario import build_scenario


def test_scenario_contract_serializes_future_multimodal_sources() -> None:
    result = build_scenario(
        "High temperature is expected tomorrow.",
        "energy_demand",
        24,
        "hours",
        "data/demo_energy_demand.csv",
        ["timeseries", "text", "image"],
    )
    assert result.as_dict() == {
        "description": "High temperature is expected tomorrow.",
        "target": "energy_demand",
        "horizon": 24,
        "time_unit": "hours",
        "dataset": "data\\demo_energy_demand.csv",
        "evidence_sources": ["timeseries", "text", "image"],
    }


def test_scenario_contract_requires_time_series_and_valid_horizon() -> None:
    with pytest.raises(ValueError, match="time-series"):
        build_scenario("Forecast demand", "demand", 3, "hours", "data.csv", ["image"])
    with pytest.raises(ValueError, match="at least 1"):
        build_scenario("Forecast demand", "demand", 0, "hours", "data.csv")
