"""Dependency-free regression metrics for forecasting experiments."""

from __future__ import annotations

import math


def _paired(actual: list[float], predicted: list[float]) -> list[tuple[float, float]]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must have the same non-zero length")
    return list(zip(actual, predicted))


def mae(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute error."""
    pairs = _paired(actual, predicted)
    return sum(abs(a - p) for a, p in pairs) / len(pairs)


def rmse(actual: list[float], predicted: list[float]) -> float:
    """Root mean squared error."""
    pairs = _paired(actual, predicted)
    return math.sqrt(sum((a - p) ** 2 for a, p in pairs) / len(pairs))


def mape(actual: list[float], predicted: list[float]) -> float:
    """Mean absolute percentage error, ignoring zero-valued actual observations."""
    pairs = [(a, p) for a, p in _paired(actual, predicted) if a != 0]
    if not pairs:
        raise ValueError("MAPE is undefined when all actual values are zero")
    return sum(abs((a - p) / a) for a, p in pairs) * 100 / len(pairs)
