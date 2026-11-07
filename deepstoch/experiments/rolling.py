"""Rolling-origin evaluation for dissertation-scale temporal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from deepstoch.experiments.hybrid import HybridScore, evaluate_ml_state_hybrid
from deepstoch.data.sequences import build_sequences_from_rows
from deepstoch.experiments.neural_eval import NeuralEvaluation, evaluate_neural_state_space
from deepstoch.models.neural_state import train_neural_state_space


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


@dataclass(frozen=True)
class RepeatedRollingSummary:
    model: str
    seeds: list[int]
    folds_per_seed: int
    test_size: int
    mean_rmse: float
    std_rmse: float
    mean_coverage: float
    std_coverage: float
    mean_interval_width: float
    seed_summaries: list[dict[str, object]]

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


def evaluate_rolling_neural_state(
    rows: list[dict[str, str]],
    target_column: str = "target",
    time_column: str = "timestamp",
    *,
    sequence_length: int = 6,
    minimum_train: int = 120,
    test_size: int = 24,
    step: int = 24,
    epochs: int = 20,
    patience: int = 5,
    nominal_coverage: float = 0.9,
    seed: int = 42,
) -> RollingSummary:
    """Retrain and evaluate the neural state-space model at each time origin.

    Each fold trains only on rows before its origin.  The validation split is
    chronological within that training prefix, and the following test block is
    kept untouched until inference.  Temporary checkpoints ensure that fold
    normalization statistics cannot leak between origins.
    """
    if sequence_length < 1 or minimum_train <= sequence_length + 1:
        raise ValueError("minimum-train must exceed sequence-length by at least two rows")
    if test_size < 1 or step < 1 or epochs < 1 or patience < 1:
        raise ValueError("test-size, step, epochs, and patience must be positive")
    if minimum_train + test_size > len(rows):
        raise ValueError("minimum_train plus test-size must fit within the dataset")
    if not 0 < nominal_coverage < 1:
        raise ValueError("coverage must be between 0 and 1")

    scores: list[NeuralEvaluation] = []
    with TemporaryDirectory(prefix="deepstoch-rolling-neural-") as temporary_directory:
        for fold, origin in enumerate(range(minimum_train, len(rows) - test_size + 1, step)):
            training_rows = rows[:origin]
            evaluation_rows = rows[:origin + test_size]
            training_batch = build_sequences_from_rows(training_rows, target_column, time_column, sequence_length)
            training_sequences, validation_sequences = training_batch.split()
            checkpoint = Path(temporary_directory) / f"fold-{fold}.pt"
            train_neural_state_space(
                training_sequences,
                input_size=len(training_sequences.sequences[0][0]),
                validation_batch=validation_sequences,
                epochs=epochs,
                patience=patience,
                checkpoint_path=checkpoint,
                seed=seed + fold,
            )
            scores.append(
                evaluate_neural_state_space(
                    evaluation_rows,
                    checkpoint,
                    target_column,
                    time_column,
                    sequence_length,
                    test_size,
                    nominal_coverage,
                )
            )
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
        sum(score.empirical_coverage for score in scores) / len(scores),
        sum(score.mean_interval_width for score in scores) / len(scores),
        [score.as_dict() for score in scores],
    )


def evaluate_repeated_rolling_neural_state(
    rows: list[dict[str, str]],
    target_column: str = "target",
    time_column: str = "timestamp",
    *,
    seeds: list[int] | tuple[int, ...] = (42, 43, 44),
    sequence_length: int = 6,
    minimum_train: int = 120,
    test_size: int = 24,
    step: int = 24,
    epochs: int = 20,
    patience: int = 5,
    nominal_coverage: float = 0.9,
) -> RepeatedRollingSummary:
    """Repeat rolling neural evaluation for independent random seeds."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    summaries = [
        evaluate_rolling_neural_state(
            rows,
            target_column,
            time_column,
            sequence_length=sequence_length,
            minimum_train=minimum_train,
            test_size=test_size,
            step=step,
            epochs=epochs,
            patience=patience,
            nominal_coverage=nominal_coverage,
            seed=seed,
        )
        for seed in seeds
    ]
    fold_scores = [fold for summary in summaries for fold in summary.fold_scores]
    rmses = [float(fold["rmse"]) for fold in fold_scores]
    coverages = [float(fold["empirical_coverage"]) for fold in fold_scores]
    mean_rmse = sum(rmses) / len(rmses)
    mean_coverage = sum(coverages) / len(coverages)
    rmse_variance = sum((value - mean_rmse) ** 2 for value in rmses) / len(rmses)
    coverage_variance = sum((value - mean_coverage) ** 2 for value in coverages) / len(coverages)
    return RepeatedRollingSummary(
        summaries[0].model,
        list(seeds),
        summaries[0].folds,
        test_size,
        mean_rmse,
        math.sqrt(rmse_variance),
        mean_coverage,
        math.sqrt(coverage_variance),
        sum(float(fold["mean_interval_width"]) for fold in fold_scores) / len(fold_scores),
        [summary.as_dict() for summary in summaries],
    )
