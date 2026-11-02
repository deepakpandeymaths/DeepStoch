"""Dependency-free split-conformal prediction intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ConformalInterval:
    nominal_coverage: float
    calibration_size: int
    radius: float
    test_size: int
    empirical_coverage: float
    mean_width: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def split_conformal_interval(
    calibration_actual: list[float],
    calibration_predicted: list[float],
    test_actual: list[float],
    test_predicted: list[float],
    *,
    nominal_coverage: float = 0.9,
) -> ConformalInterval:
    """Calibrate a constant-radius interval from absolute calibration residuals."""
    if len(calibration_actual) != len(calibration_predicted) or not calibration_actual:
        raise ValueError("calibration observations must be non-empty and paired")
    if len(test_actual) != len(test_predicted) or not test_actual:
        raise ValueError("test observations must be non-empty and paired")
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between 0 and 1")
    scores = sorted(abs(actual - predicted) for actual, predicted in zip(calibration_actual, calibration_predicted))
    rank = math.ceil((len(scores) + 1) * nominal_coverage) - 1
    radius = scores[min(rank, len(scores) - 1)]
    covered = sum(abs(actual - predicted) <= radius for actual, predicted in zip(test_actual, test_predicted))
    return ConformalInterval(nominal_coverage, len(scores), radius, len(test_actual), covered / len(test_actual), 2 * radius)
