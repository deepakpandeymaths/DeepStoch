"""Validated scenario contracts for multimodal forecasting extensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ScenarioSpec:
    """A forecast question plus the evidence sources allowed for a run."""

    description: str
    target: str
    horizon: int
    time_unit: str
    dataset: str
    evidence_sources: tuple[str, ...] = field(default_factory=lambda: ("timeseries",))

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"evidence_sources": list(self.evidence_sources)}


def build_scenario(
    description: str,
    target: str,
    horizon: int,
    time_unit: str,
    dataset: str,
    evidence_sources: list[str] | tuple[str, ...] = ("timeseries",),
) -> ScenarioSpec:
    """Build a scenario after validating its forecasting contract."""
    clean_description = description.strip()
    clean_target = target.strip()
    clean_unit = time_unit.strip().lower()
    clean_dataset = str(Path(dataset.strip()))
    allowed_sources = {"timeseries", "text", "image", "document", "video"}
    clean_sources = tuple(dict.fromkeys(source.strip().lower() for source in evidence_sources if source.strip()))
    if not clean_description:
        raise ValueError("scenario description must not be empty")
    if not clean_target:
        raise ValueError("scenario target must not be empty")
    if horizon < 1:
        raise ValueError("scenario horizon must be at least 1")
    if not clean_unit:
        raise ValueError("scenario time-unit must not be empty")
    if not dataset.strip():
        raise ValueError("scenario dataset must not be empty")
    if not clean_sources:
        raise ValueError("at least one evidence source is required")
    unknown = set(clean_sources) - allowed_sources
    if unknown:
        raise ValueError(f"unsupported evidence source: {sorted(unknown)[0]}")
    if "timeseries" not in clean_sources:
        raise ValueError("the first scenario contract requires a time-series source")
    return ScenarioSpec(clean_description, clean_target, horizon, clean_unit, clean_dataset, clean_sources)
