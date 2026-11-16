"""Structured scenario specifications for future multimodal forecasting."""

from deepstoch.scenario.spec import ScenarioSpec, build_scenario
from deepstoch.scenario.context import TextContext, extract_text_context
from deepstoch.scenario.registry import EvidenceFeature, FeatureRegistry, context_to_registry

__all__ = ["EvidenceFeature", "FeatureRegistry", "ScenarioSpec", "TextContext", "build_scenario", "context_to_registry", "extract_text_context"]
