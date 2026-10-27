"""Chronological train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[T, ...]
    validation: tuple[T, ...]
    test: tuple[T, ...]


def chronological_split(
    rows: list[T], *, train_fraction: float = 0.7, validation_fraction: float = 0.15
) -> TemporalSplit:
    """Split ordered rows without shuffling future observations into training."""

    if not rows:
        raise ValueError("Cannot split an empty dataset")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    total = len(rows)
    train_end = max(1, int(total * train_fraction))
    validation_end = max(train_end, int(total * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, total - 1) if total > 1 else total

    return TemporalSplit(tuple(rows[:train_end]), tuple(rows[train_end:validation_end]), tuple(rows[validation_end:]))

