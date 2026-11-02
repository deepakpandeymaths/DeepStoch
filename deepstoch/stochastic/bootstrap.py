"""Reproducible residual-bootstrap trajectory simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class SimulationSummary:
    paths: int
    horizon: int
    quantiles: dict[str, list[float]]
    threshold: float | None
    probability_exceeds_threshold: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def simulate_bootstrap_paths(history: list[float], innovations: list[float], horizon: int, paths: int = 1000, seed: int = 42) -> list[list[float]]:
    """Generate random-walk trajectories by resampling historical innovations."""
    if len(history) < 2:
        raise ValueError("history must contain at least two observations")
    if not innovations:
        raise ValueError("innovations must contain at least one value")
    if horizon < 1 or paths < 1:
        raise ValueError("horizon and paths must be at least 1")
    generator = random.Random(seed)
    trajectories: list[list[float]] = []
    for _ in range(paths):
        current = float(history[-1])
        path = []
        for _ in range(horizon):
            current += generator.choice(innovations)
            path.append(current)
        trajectories.append(path)
    return trajectories


def summarize_paths(trajectories: list[list[float]], *, threshold: float | None = None) -> SimulationSummary:
    """Summarize simulated paths at each future step."""
    if not trajectories or not trajectories[0]:
        raise ValueError("trajectories must be non-empty")
    horizon = len(trajectories[0])
    if any(len(path) != horizon for path in trajectories):
        raise ValueError("all trajectories must have the same horizon")

    def quantile(values: list[float], level: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * level
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    columns = [[path[step] for path in trajectories] for step in range(horizon)]
    quantiles = {str(level): [quantile(column, level) for column in columns] for level in (0.1, 0.5, 0.9)}
    exceedance = None
    if threshold is not None:
        exceedance = sum(any(value >= threshold for value in path) for path in trajectories) / len(trajectories)
    return SimulationSummary(len(trajectories), horizon, quantiles, threshold, exceedance)
