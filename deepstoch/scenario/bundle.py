"""Unified scenario and evidence bundle for future forecasting models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from deepstoch.data.loader import DataLoadError, load_csv
from deepstoch.data.split import chronological_split
from deepstoch.data.validation import ValidationIssue, validate_rows
from deepstoch.scenario.context import extract_text_context
from deepstoch.scenario.document import extract_document_evidence
from deepstoch.scenario.image import extract_image_evidence
from deepstoch.scenario.registry import EvidenceFeature, FeatureRegistry
from deepstoch.scenario.spec import ScenarioSpec


@dataclass(frozen=True)
class EvidenceBundle:
    scenario: dict[str, object]
    dataset_readiness: dict[str, object]
    attached_sources: list[str]
    feature_count: int
    features: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetReadiness:
    path: str
    valid: bool
    row_count: int
    columns: list[str]
    target_column: str
    time_column: str
    horizon: int
    minimum_rows: int
    errors: list[dict[str, object]]
    warnings: list[dict[str, object]]
    split_sizes: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _issue_dict(issue: ValidationIssue) -> dict[str, object]:
    return asdict(issue)


def validate_scenario_dataset(scenario: ScenarioSpec, *, time_column: str = "timestamp") -> DatasetReadiness:
    """Check that a scenario target and horizon fit its selected CSV dataset."""
    minimum_rows = scenario.horizon + 3
    try:
        headers, rows = load_csv(Path(scenario.dataset))
    except DataLoadError as exc:
        return DatasetReadiness(
            str(scenario.dataset), False, 0, [], scenario.target, time_column,
            scenario.horizon, minimum_rows,
            [{"code": "load-error", "message": str(exc), "row": None, "column": None}],
            [], {"train": 0, "validation": 0, "test": 0},
        )
    validation = validate_rows(headers, rows, time_column=time_column, target_column=scenario.target)
    errors = [_issue_dict(issue) for issue in validation.errors]
    warnings = [_issue_dict(issue) for issue in validation.warnings]
    if len(rows) < minimum_rows:
        errors.append({
            "code": "insufficient-history",
            "message": f"Dataset has {len(rows)} rows; at least {minimum_rows} are required for horizon {scenario.horizon}",
            "row": None,
            "column": None,
        })
    split_sizes = {"train": 0, "validation": 0, "test": 0}
    if rows:
        split = chronological_split(rows)
        split_sizes = {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)}
    return DatasetReadiness(
        str(scenario.dataset), not errors, len(rows), headers, scenario.target, time_column,
        scenario.horizon, minimum_rows, errors, warnings, split_sizes,
    )


def _add_features(registry: FeatureRegistry, features: list[dict[str, object]]) -> None:
    for feature in features:
        registry.add(EvidenceFeature(**feature))


def build_evidence_bundle(
    scenario: ScenarioSpec,
    *,
    text: str | None = None,
    documents: tuple[str | Path, ...] = (),
    images: tuple[str | Path, ...] = (),
    time_column: str = "timestamp",
) -> EvidenceBundle:
    """Combine declared scenario evidence into one validated feature bundle."""
    readiness = validate_scenario_dataset(scenario, time_column=time_column)
    if not readiness.valid:
        raise ValueError("scenario dataset is not ready: " + "; ".join(item["message"] for item in readiness.errors))
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
    return EvidenceBundle(scenario.as_dict(), readiness.as_dict(), attached, len(registry), registry.as_dict())
