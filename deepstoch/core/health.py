"""Dependency-free project health checks."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path

from deepstoch import __version__


@dataclass(frozen=True)
class HealthReport:
    status: str
    project: str
    version: str
    python: str
    platform: str
    repository_root: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def find_repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def run_health_check(start: Path | None = None) -> HealthReport:
    root = find_repository_root(start)
    return HealthReport(
        status="ok",
        project="DeepStoch",
        version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        repository_root=str(root),
    )

