"""Unified chronological model comparison report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from deepstoch.data.features import build_features
from deepstoch.evaluation.metrics import mae, mape, rmse
from deepstoch.experiments.baselines import evaluate_baselines
from deepstoch.experiments.hybrid import evaluate_local_level, evaluate_local_trend, evaluate_ml_state_hybrid
from deepstoch.experiments.ml import evaluate_linear_regression, evaluate_random_forest
from deepstoch.experiments.neural_eval import evaluate_neural_state_space


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    mae: float | None
    rmse: float | None
    mape: float | None
    coverage: float | None
    interval_width: float | None
    status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compare_models(rows: list[dict[str, str]], *, target_column: str = "target", time_column: str = "timestamp", test_size: int = 8, neural_checkpoint: str | Path | None = None) -> list[ComparisonRow]:
    """Evaluate all available models against one final chronological holdout."""
    values = [float(row[target_column]) for row in rows]
    if test_size < 1 or test_size >= len(values):
        raise ValueError("test-size must be at least 1 and smaller than the dataset")
    scores = evaluate_baselines(values[:-test_size], values[-test_size:], test_size)
    result = [ComparisonRow(score.model, score.mae, score.rmse, score.mape, None, None, "ok") for score in scores]
    result.extend(_point_row(evaluate_linear_regression(rows, target_column, time_column, test_size)))
    result.extend(_point_row(evaluate_random_forest(rows, target_column, time_column, test_size)))
    for score in (evaluate_local_level(rows, target_column, test_size), evaluate_local_trend(rows, target_column, test_size), evaluate_ml_state_hybrid(rows, target_column, time_column, test_size)):
        result.append(ComparisonRow(score.model, score.mae, score.rmse, None, score.interval_coverage, score.mean_interval_width, "ok"))
    if neural_checkpoint is not None and Path(neural_checkpoint).exists():
        score = evaluate_neural_state_space(rows, neural_checkpoint, target_column, time_column, 6, test_size)
        result.append(ComparisonRow(score.model, score.mae, score.rmse, score.mape, score.empirical_coverage, score.mean_interval_width, "ok"))
    return sorted(result, key=lambda row: row.rmse if row.rmse is not None else float("inf"))


def _point_row(score) -> list[ComparisonRow]:
    return [ComparisonRow(score.model, score.mae, score.rmse, score.mape, None, None, "ok")]
