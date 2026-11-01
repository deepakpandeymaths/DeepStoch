"""Leakage-safe time-series feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math


@dataclass(frozen=True)
class FeatureRow:
    index: int
    features: list[float]
    target: float


FEATURE_NAMES = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "hour_sin", "hour_cos", "temperature", "holiday"]


def build_features(rows: list[dict[str, str]], target_column: str = "target", time_column: str = "timestamp") -> list[FeatureRow]:
    """Build rows using only current metadata and prior target observations."""
    if len(rows) < 4:
        raise ValueError("at least four ordered rows are required for lag features")
    targets = [float(row[target_column]) for row in rows]
    output: list[FeatureRow] = []
    for index in range(3, len(rows)):
        timestamp = _parse_timestamp(rows[index][time_column])
        history = targets[index - 3:index]
        temperature = float(rows[index].get("temperature", "0") or 0)
        holiday = float(rows[index].get("holiday", "0") or 0)
        hour = timestamp.hour + timestamp.minute / 60
        output.append(FeatureRow(index, [history[-1], history[-2], history[-3], sum(history) / 3, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24), temperature, holiday], targets[index]))
    return output


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value}") from exc
