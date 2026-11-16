"""Transparent text-context extraction for future multimodal scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


SIGNAL_TERMS: dict[str, tuple[str, ...]] = {
    "heat": ("heat", "hot", "high temperature", "heatwave", "heat wave", "summer"),
    "cold": ("cold", "cool", "low temperature", "winter", "freezing"),
    "maintenance": ("maintenance", "repair", "service", "故障", "fault"),
    "occupancy": ("occupancy", "occupied", "crowd", "people", "attendance"),
    "weather": ("weather", "rain", "storm", "wind", "humidity", "temperature"),
    "event": ("event", "holiday", "festival", "schedule", "meeting"),
}


@dataclass(frozen=True)
class TextContext:
    normalized_text: str
    token_count: int
    detected_signals: dict[str, float]
    matched_terms: dict[str, list[str]]
    numeric_mentions: list[float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_text_context(description: str) -> TextContext:
    """Extract auditable keyword and numeric signals without generative inference."""
    normalized = " ".join(description.lower().split())
    if not normalized:
        raise ValueError("description must not be empty")
    matched: dict[str, list[str]] = {}
    signals: dict[str, float] = {}
    for signal, terms in SIGNAL_TERMS.items():
        found = [term for term in terms if term in normalized]
        if found:
            matched[signal] = found
            signals[signal] = 1.0
    numbers = [float(value) for value in re.findall(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", normalized)]
    return TextContext(normalized, len(normalized.split()), signals, matched, numbers)
