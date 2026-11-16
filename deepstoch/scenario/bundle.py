"""Unified scenario and evidence bundle for future forecasting models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from deepstoch.scenario.context import extract_text_context
from deepstoch.scenario.document import extract_document_evidence
from deepstoch.scenario.image import extract_image_evidence
from deepstoch.scenario.registry import EvidenceFeature, FeatureRegistry
from deepstoch.scenario.spec import ScenarioSpec


@dataclass(frozen=True)
class EvidenceBundle:
    scenario: dict[str, object]
    attached_sources: list[str]
    feature_count: int
    features: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _add_features(registry: FeatureRegistry, features: list[dict[str, object]]) -> None:
    for feature in features:
        registry.add(EvidenceFeature(**feature))


def build_evidence_bundle(
    scenario: ScenarioSpec,
    *,
    text: str | None = None,
    documents: tuple[str | Path, ...] = (),
    images: tuple[str | Path, ...] = (),
) -> EvidenceBundle:
    """Combine declared scenario evidence into one validated feature bundle."""
    registry = FeatureRegistry()
    attached: list[str] = ["timeseries"]
    if text is not None:
        if "text" not in scenario.evidence_sources:
            raise ValueError("text evidence was supplied but text is not declared")
        context = extract_text_context(text)
        from deepstoch.scenario.registry import context_to_registry

        _add_features(registry, context_to_registry(context, source_ref="scenario.text").as_dict())
        attached.append("text")
    if documents:
        if "document" not in scenario.evidence_sources:
            raise ValueError("document evidence was supplied but document is not declared")
        for path in documents:
            _add_features(registry, extract_document_evidence(path).features)
        attached.append("document")
    if images:
        if "image" not in scenario.evidence_sources:
            raise ValueError("image evidence was supplied but image is not declared")
        for path in images:
            _add_features(registry, extract_image_evidence(path).features)
        attached.append("image")
    return EvidenceBundle(scenario.as_dict(), attached, len(registry), registry.as_dict())
