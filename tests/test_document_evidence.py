import json

import pytest

from deepstoch.scenario import extract_document_evidence


def test_document_features_keep_file_provenance(tmp_path) -> None:
    path = tmp_path / "brief.md"
    path.write_text("Heat event expected at 35 degrees.", encoding="utf-8")

    result = extract_document_evidence(path)

    assert result.document_type == "md"
    assert result.character_count > 0
    assert result.features[0]["source_type"] == "document"
    assert result.features[0]["source_ref"] == str(path)
    assert any(feature["value"] == 35.0 for feature in result.features)


def test_json_documents_are_validated(tmp_path) -> None:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({"description": "Rain expected at 12 mm"}), encoding="utf-8")

    result = extract_document_evidence(path)

    assert result.document_type == "json"
    assert result.context["numeric_mentions"] == [12.0]


def test_document_adapter_rejects_invalid_or_unsupported_files(tmp_path) -> None:
    invalid = tmp_path / "broken.json"
    invalid.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid"):
        extract_document_evidence(invalid)

    image = tmp_path / "photo.png"
    image.write_bytes(b"not an image processed by this adapter")
    with pytest.raises(ValueError, match="unsupported document type"):
        extract_document_evidence(image)
