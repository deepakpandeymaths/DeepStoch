import pytest

from deepstoch.scenario import build_evidence_bundle, build_scenario


def test_bundle_combines_declared_text_and_document_evidence(tmp_path) -> None:
    document = tmp_path / "notes.md"
    document.write_text("Rain expected at 12 mm", encoding="utf-8")
    scenario = build_scenario("Demand tomorrow", "load", 24, "hours", "data.csv", ["timeseries", "text", "document"])

    result = build_evidence_bundle(scenario, text="Heat expected at 35 degrees", documents=(document,))

    assert result.feature_count == len(result.features)
    assert result.attached_sources == ["timeseries", "text", "document"]
    assert {feature["source_type"] for feature in result.features} == {"text", "document"}
    assert any(feature["source_ref"] == str(document) for feature in result.features)


def test_bundle_requires_declared_source_for_attached_evidence(tmp_path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("Heat at 35 degrees", encoding="utf-8")
    scenario = build_scenario("Demand tomorrow", "load", 24, "hours", "data.csv")

    with pytest.raises(ValueError, match="document is not declared"):
        build_evidence_bundle(scenario, documents=(document,))
