"""Inference and test-set evaluation for neural state-space checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from deepstoch.data.sequences import build_sequences_from_rows
from deepstoch.evaluation.metrics import mae, mape, rmse
from deepstoch.models.neural_state import _torch, create_neural_state_space


@dataclass(frozen=True)
class NeuralEvaluation:
    model: str
    test_size: int
    mae: float
    rmse: float
    mape: float
    nominal_coverage: float
    empirical_coverage: float
    mean_interval_width: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_neural_state_space(
    rows: list[dict[str, str]],
    checkpoint_path: str | Path,
    target_column: str = "target",
    time_column: str = "timestamp",
    sequence_length: int = 6,
    test_size: int = 8,
    nominal_coverage: float = 0.9,
) -> NeuralEvaluation:
    """Load a checkpoint and evaluate only the final chronological sequences."""
    torch, _ = _torch()
    batch = build_sequences_from_rows(rows, target_column, time_column, sequence_length)
    if test_size < 1 or test_size >= len(batch.sequences):
        raise ValueError("test-size must be at least 1 and smaller than the usable sequence count")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    if not 0 < nominal_coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    saved = torch.load(checkpoint, map_location="cpu")
    input_size = int(saved.get("input_size", len(batch.sequences[0][0])))
    model = create_neural_state_space(input_size)
    model.load_state_dict(saved["model_state"])
    model.eval()
    feature_mean = saved.get("feature_mean", [0.0] * input_size)
    feature_std = saved.get("feature_std", [1.0] * input_size)
    target_mean = float(saved.get("target_mean", 0.0))
    target_std = float(saved.get("target_std", 1.0))
    normalized = [[[(value - feature_mean[column]) / (feature_std[column] or 1.0) for column, value in enumerate(row)] for row in sequence] for sequence in batch.sequences[-test_size:]]
    features = torch.tensor(normalized, dtype=torch.float32)
    actual = batch.targets[-test_size:]
    with torch.no_grad():
        mean_tensor, scale_tensor = model(features)
    predicted = [target_mean + target_std * float(value) for value in mean_tensor.tolist()]
    scales = [target_std * float(value) for value in scale_tensor.tolist()]
    z = 1.645 if nominal_coverage == 0.9 else _normal_critical_value(nominal_coverage)
    lower = [prediction - z * scale for prediction, scale in zip(predicted, scales)]
    upper = [prediction + z * scale for prediction, scale in zip(predicted, scales)]
    covered = sum(low <= value <= high for value, low, high in zip(actual, lower, upper))
    return NeuralEvaluation("neural_state_space", test_size, mae(actual, predicted), rmse(actual, predicted), mape(actual, predicted), nominal_coverage, covered / test_size, sum(high - low for low, high in zip(lower, upper)) / test_size)


def _normal_critical_value(coverage: float) -> float:
    """Approximate the normal critical value without requiring SciPy."""
    if coverage <= 0.8:
        return 1.282
    if coverage <= 0.9:
        return 1.645
    if coverage <= 0.95:
        return 1.960
    if coverage <= 0.99:
        return 2.576
    return 3.291
