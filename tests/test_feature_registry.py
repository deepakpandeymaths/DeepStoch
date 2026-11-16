import pytest

from deepstoch.scenario import EvidenceFeature, FeatureRegistry, context_to_registry, extract_text_context


def test_context_features_preserve_source_and_transformation() -> None:
    registry = context_to_registry(extract_text_context("Heat expected at 35 degrees"))
    features = registry.as_dict()
    assert len(features) == 2
    assert features[0]["source_type"] == "text"
    assert features[0]["source_ref"] == "scenario.description"
    assert features[0]["transformation"] == "keyword_presence"
    assert features[1]["transformation"] == "regex_number_extraction"


def test_feature_registry_rejects_invalid_provenance() -> None:
    registry = FeatureRegistry()
    with pytest.raises(ValueError, match="confidence"):
        registry.add(EvidenceFeature("heat", 1, "text", "description", None, 1.2, "keyword_presence"))
    with pytest.raises(ValueError, match="ISO-8601"):
        registry.add(EvidenceFeature("heat", 1, "text", "description", "tomorrow", 1, "keyword_presence"))
