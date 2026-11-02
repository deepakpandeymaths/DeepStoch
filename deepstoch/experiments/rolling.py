"""Rolling-origin evaluation for dissertation-scale temporal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from deepstoch.experiments.hybrid import HybridScore, evaluate_ml_state_hybrid


@dataclass(frozen=True)
class RollingSummary:
    model: str
    folds: int
    test_size: int
    mean_rmse: float
    std_rmse: float
    mean_coverage: float
    mean_interval_width: float
    fold_scores: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_rolling_ml_state(
    rows: list[dict[str, str]],
    target_column: str = "target",
    time_column: str = "timestamp",
    *,
    minimum_train: int = 24,
    test_size: int = 4,
    step: int = 4,
) -> RollingSummary:
    """Evaluate the ML-state hybrid repeatedly without using future rows."""
    if minimum_train < 4 or test_size < 1 or step < 1:
        raise ValueError("minimum_train must be at least 4; test-size and step must be positive")
    if minimum_train + test_size > len(rows):
        raise ValueError("minimum_train plus test-size must fit within the dataset")
    scores: list[HybridScore] = []
    for origin in range(minimum_train, len(rows) - test_size + 1, step):
        window = rows[:origin + test_size]
        scores.append(evaluate_ml_state_hybrid(window, target_column, time_column, test_size))
    if not scores:
        raise ValueError("rolling configuration produced no evaluation folds")
    rmses = [score.rmse for score in scores]
    mean_rmse = sum(rmses) / len(rmses)
    variance = sum((value - mean_rmse) ** 2 for value in rmses) / len(rmses)
    return RollingSummary(
        scores[0].model,
        len(scores),
        test_size,
        mean_rmse,
        math.sqrt(variance),
        sum(score.interval_coverage for score in scores) / len(scores),
        sum(score.mean_interval_width for score in scores) / len(scores),
        [score.as_dict() for score in scores],
    )
