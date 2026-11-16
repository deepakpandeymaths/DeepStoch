"""Structured scenario specifications for future multimodal forecasting."""

from deepstoch.scenario.spec import ScenarioSpec, build_scenario
from deepstoch.scenario.context import TextContext, extract_text_context
from deepstoch.scenario.bundle import DatasetReadiness, EvidenceBundle, build_evidence_bundle, validate_scenario_dataset
from deepstoch.scenario.document import DocumentEvidence, extract_document_evidence
from deepstoch.scenario.image import ImageEvidence, extract_image_evidence
from deepstoch.scenario.registry import EvidenceFeature, FeatureRegistry, context_to_registry

__all__ = [
    "DocumentEvidence",
    "DatasetReadiness",
    "EvidenceBundle",
    "EvidenceFeature",
    "FeatureRegistry",
    "ImageEvidence",
    "ScenarioSpec",
    "TextContext",
    "build_scenario",
    "build_evidence_bundle",
    "validate_scenario_dataset",
    "context_to_registry",
    "extract_document_evidence",
    "extract_image_evidence",
    "extract_text_context",
]
