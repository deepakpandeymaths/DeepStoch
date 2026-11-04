from pathlib import Path
import importlib.util

import pytest

from deepstoch.data.loader import load_csv
from deepstoch.data.sequences import build_sequences_from_rows
from deepstoch.models.neural_state import create_neural_state_space


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_neural_state_space_dependency_is_explicit_when_torch_is_unavailable() -> None:
    _, rows = load_csv(DEMO)
    batch = build_sequences_from_rows(rows, sequence_length=4)
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(RuntimeError, match="deep-learning"):
            create_neural_state_space(len(batch.sequences[0][0]))
    else:
        model = create_neural_state_space(len(batch.sequences[0][0]))
        assert model is not None
