"""Chronological sequence construction for neural forecasting."""

from __future__ import annotations

from dataclasses import dataclass

from deepstoch.data.features import FeatureRow, build_features


@dataclass(frozen=True)
class SequenceBatch:
    sequences: list[list[list[float]]]
    targets: list[float]
    end_indices: list[int]


def build_sequences(feature_rows: list[FeatureRow], sequence_length: int = 6) -> SequenceBatch:
    """Create past-only feature windows whose label is the next row target."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if len(feature_rows) <= sequence_length:
        raise ValueError("more feature rows than sequence_length are required")
    sequences = []
    targets = []
    end_indices = []
    for end in range(sequence_length, len(feature_rows)):
        window = [row.features[:] for row in feature_rows[end - sequence_length:end]]
        sequences.append(window)
        targets.append(feature_rows[end].target)
        end_indices.append(feature_rows[end].index)
    return SequenceBatch(sequences, targets, end_indices)


def build_sequences_from_rows(rows: list[dict[str, str]], target_column: str = "target", time_column: str = "timestamp", sequence_length: int = 6) -> SequenceBatch:
    """Build neural sequences directly from validated CSV rows."""
    return build_sequences(build_features(rows, target_column, time_column), sequence_length)
