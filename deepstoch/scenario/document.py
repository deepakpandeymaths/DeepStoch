"""Deterministic local document evidence extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from deepstoch.scenario.context import extract_text_context
from deepstoch.scenario.registry import context_to_registry


SUPPORTED_DOCUMENTS = {".txt", ".md", ".markdown", ".log", ".json"}


@dataclass(frozen=True)
class DocumentEvidence:
    path: str
    document_type: str
    character_count: int
    context: dict[str, object]
    features: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_document_evidence(path: str | Path) -> DocumentEvidence:
    """Read a supported local document and extract auditable features."""
    document_path = Path(path)
    if not document_path.is_file():
        raise ValueError(f"document does not exist: {document_path}")
    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENTS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENTS))
        raise ValueError(f"unsupported document type: {suffix or '<none>'}; use {supported}")
    try:
        content = document_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("document must be UTF-8 text") from exc
    if suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON document is not valid") from exc
    context = extract_text_context(content)
    registry = context_to_registry(context, source_ref=str(document_path), source_type="document")
    return DocumentEvidence(
        path=str(document_path),
        document_type=suffix.lstrip("."),
        character_count=len(content),
        context=context.as_dict(),
        features=registry.as_dict(),
    )
