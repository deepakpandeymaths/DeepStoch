"""Deterministic forecasting baselines used as Phase 2 reference models."""

from __future__ import annotations


def _check_history(history: list[float], horizon: int) -> None:
    if not history:
        raise ValueError("history must contain at least one value")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")


def last_value_forecast(history: list[float], horizon: int) -> list[float]:
    """Repeat the most recently observed value."""
    _check_history(history, horizon)
    return [float(history[-1])] * horizon


def mean_forecast(history: list[float], horizon: int) -> list[float]:
    """Repeat the arithmetic mean of the observed history."""
    _check_history(history, horizon)
    value = sum(history) / len(history)
    return [float(value)] * horizon


def moving_average_forecast(history: list[float], horizon: int, window: int = 3) -> list[float]:
    """Forecast by repeating the mean of the most recent ``window`` values."""
    _check_history(history, horizon)
    if window < 1:
        raise ValueError("window must be at least 1")
    recent = history[-window:]
    value = sum(recent) / len(recent)
    return [float(value)] * horizon


def seasonal_naive_forecast(history: list[float], horizon: int, season_length: int = 24) -> list[float]:
    """Repeat the latest complete seasonal cycle, with a safe short-history fallback."""
    _check_history(history, horizon)
    if season_length < 1:
        raise ValueError("season_length must be at least 1")
    if len(history) < season_length:
        return last_value_forecast(history, horizon)
    cycle = history[-season_length:]
    return [float(cycle[index % season_length]) for index in range(horizon)]
