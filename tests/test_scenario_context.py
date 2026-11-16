import pytest

from deepstoch.scenario import extract_text_context


def test_text_context_extraction_is_transparent_and_auditable() -> None:
    result = extract_text_context("A heat wave and high temperature are expected during a scheduled event at 35 degrees.")
    assert result.token_count > 0
    assert result.detected_signals["heat"] == 1.0
    assert result.detected_signals["weather"] == 1.0
    assert "high temperature" in result.matched_terms["heat"]
    assert result.numeric_mentions == [35.0]


def test_text_context_requires_non_empty_description() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_text_context("  ")
