"""Structured scenario specifications for future multimodal forecasting."""

from deepstoch.scenario.spec import ScenarioSpec, build_scenario
from deepstoch.scenario.context import TextContext, extract_text_context
from deepstoch.scenario.document import DocumentEvidence, extract_document_evidence
from deepstoch.scenario.image import ImageEvidence, extract_image_evidence
from deepstoch.scenario.registry import EvidenceFeature, FeatureRegistry, context_to_registry

__all__ = [
    "DocumentEvidence",
    "EvidenceFeature",
    "FeatureRegistry",
    "ImageEvidence",
    "ScenarioSpec",
    "TextContext",
    "build_scenario",
    "context_to_registry",
    "extract_document_evidence",
    "extract_image_evidence",
    "extract_text_context",
]
