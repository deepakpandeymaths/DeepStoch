import pytest

from deepstoch.scenario import build_evidence_bundle, build_scenario


def test_bundle_combines_declared_text_and_document_evidence(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("timestamp,load\n2026-01-01T00:00:00,10\n2026-01-01T01:00:00,11\n2026-01-01T02:00:00,12\n2026-01-01T03:00:00,13\n2026-01-01T04:00:00,14\n2026-01-01T05:00:00,15\n2026-01-01T06:00:00,16\n2026-01-01T07:00:00,17\n2026-01-01T08:00:00,18\n2026-01-01T09:00:00,19\n2026-01-01T10:00:00,20\n2026-01-01T11:00:00,21\n2026-01-01T12:00:00,22\n2026-01-01T13:00:00,23\n2026-01-01T14:00:00,24\n2026-01-01T15:00:00,25\n2026-01-01T16:00:00,26\n2026-01-01T17:00:00,27\n2026-01-01T18:00:00,28\n2026-01-01T19:00:00,29\n2026-01-01T20:00:00,30\n2026-01-01T21:00:00,31\n2026-01-01T22:00:00,32\n2026-01-01T23:00:00,33\n", encoding="utf-8")
    document = tmp_path / "notes.md"
    document.write_text("Rain expected at 12 mm", encoding="utf-8")
    scenario = build_scenario("Demand tomorrow", "load", 12, "hours", str(dataset), ["timeseries", "text", "document"])

    result = build_evidence_bundle(scenario, text="Heat expected at 35 degrees", documents=(document,))

    assert result.feature_count == len(result.features)
    assert result.dataset_readiness["valid"] is True
    assert result.attached_sources == ["timeseries", "text", "document"]
    assert {feature["source_type"] for feature in result.features} == {"text", "document"}
    assert any(feature["source_ref"] == str(document) for feature in result.features)


def test_bundle_requires_declared_source_for_attached_evidence(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("timestamp,load\n2026-01-01T00:00:00,10\n2026-01-01T01:00:00,11\n2026-01-01T02:00:00,12\n2026-01-01T03:00:00,13\n", encoding="utf-8")
    document = tmp_path / "notes.txt"
    document.write_text("Heat at 35 degrees", encoding="utf-8")
    scenario = build_scenario("Demand tomorrow", "load", 1, "hours", str(dataset))

    with pytest.raises(ValueError, match="document is not declared"):
        build_evidence_bundle(scenario, documents=(document,))


def test_dataset_readiness_reports_missing_target(tmp_path) -> None:
    from deepstoch.scenario import validate_scenario_dataset

    dataset = tmp_path / "data.csv"
    dataset.write_text("timestamp,other\n2026-01-01T00:00:00,10\n2026-01-01T01:00:00,11\n2026-01-01T02:00:00,12\n", encoding="utf-8")
    scenario = build_scenario("Demand tomorrow", "load", 1, "hours", str(dataset))

    readiness = validate_scenario_dataset(scenario)

    assert readiness.valid is False
    assert any(error["code"] == "missing-column" for error in readiness.errors)
