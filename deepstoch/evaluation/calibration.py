"""Calibration diagnostics for probabilistic rolling forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CalibrationSummary:
    """Aggregate interval calibration diagnostics across temporal folds."""

    nominal_coverage: float
    empirical_coverage: float
    coverage_error: float
    absolute_coverage_error: float
    minimum_fold_coverage: float
    maximum_fold_coverage: float
    mean_interval_width: float
    width_standard_deviation: float
    folds: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_calibration(fold_scores: list[dict[str, object]]) -> CalibrationSummary:
    """Summarize coverage and interval width without hiding fold variation."""
    if not fold_scores:
        raise ValueError("at least one fold score is required")
    required = {"nominal_coverage", "empirical_coverage", "mean_interval_width"}
    if any(not required.issubset(score) for score in fold_scores):
        raise ValueError("fold scores must contain nominal_coverage, empirical_coverage, and mean_interval_width")
    nominal_values = {float(score["nominal_coverage"]) for score in fold_scores}
    if len(nominal_values) != 1:
        raise ValueError("all fold scores must use the same nominal coverage")
    nominal = nominal_values.pop()
    coverages = [float(score["empirical_coverage"]) for score in fold_scores]
    widths = [float(score["mean_interval_width"]) for score in fold_scores]
    empirical = sum(coverages) / len(coverages)
    mean_width = sum(widths) / len(widths)
    width_variance = sum((width - mean_width) ** 2 for width in widths) / len(widths)
    return CalibrationSummary(
        nominal,
        empirical,
        empirical - nominal,
        abs(empirical - nominal),
        min(coverages),
        max(coverages),
        mean_width,
        width_variance ** 0.5,
        len(fold_scores),
    )
