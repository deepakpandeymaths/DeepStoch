"""Phase 6 Monte Carlo simulation experiment."""

from __future__ import annotations

from deepstoch.stochastic.bootstrap import SimulationSummary, simulate_bootstrap_paths, summarize_paths


def simulate_dataset(rows: list[dict[str, str]], target_column: str = "target", horizon: int = 24, paths: int = 1000, seed: int = 42, threshold: float | None = None) -> SimulationSummary:
    """Estimate future trajectories from historical first differences."""
    history = [float(row[target_column]) for row in rows]
    if len(history) < 2:
        raise ValueError("at least two observations are required for simulation")
    innovations = [current - previous for previous, current in zip(history, history[1:])]
    trajectories = simulate_bootstrap_paths(history, innovations, horizon, paths, seed)
    return summarize_paths(trajectories, threshold=threshold)
