"""Provenance-aware feature registry for future multimodal evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from deepstoch.scenario.context import TextContext


@dataclass(frozen=True)
class EvidenceFeature:
    name: str
    value: float
    source_type: str
    source_ref: str
    timestamp: str | None
    confidence: float
    transformation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FeatureRegistry:
    """Ordered collection of evidence features with provenance checks."""

    def __init__(self) -> None:
        self._features: list[EvidenceFeature] = []

    def add(self, feature: EvidenceFeature) -> None:
        if not feature.name.strip():
            raise ValueError("feature name must not be empty")
        if feature.source_type not in {"timeseries", "text", "image", "document", "video"}:
            raise ValueError(f"unsupported feature source: {feature.source_type}")
        if not 0 <= feature.confidence <= 1:
            raise ValueError("feature confidence must be between 0 and 1")
        if feature.timestamp is not None:
            try:
                datetime.fromisoformat(feature.timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("feature timestamp must be ISO-8601") from exc
        self._features.append(feature)

    def as_dict(self) -> list[dict[str, object]]:
        return [feature.as_dict() for feature in self._features]

    def __len__(self) -> int:
        return len(self._features)


def context_to_registry(context: TextContext, *, source_ref: str = "scenario.description") -> FeatureRegistry:
    """Convert extracted text signals into provenance-aware numeric features."""
    registry = FeatureRegistry()
    for name, value in context.detected_signals.items():
        registry.add(EvidenceFeature(name, value, "text", source_ref, None, 1.0, "keyword_presence"))
    for index, value in enumerate(context.numeric_mentions):
        registry.add(EvidenceFeature(f"numeric_mention_{index + 1}", value, "text", source_ref, None, 1.0, "regex_number_extraction"))
    return registry
