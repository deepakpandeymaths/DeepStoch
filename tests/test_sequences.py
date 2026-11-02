from pathlib import Path

from deepstoch.data.loader import load_csv
from deepstoch.data.sequences import build_sequences_from_rows


DEMO = Path(__file__).parents[1] / "data" / "demo_energy_demand.csv"


def test_sequences_are_chronological_and_past_only() -> None:
    _, rows = load_csv(DEMO)
    batch = build_sequences_from_rows(rows, sequence_length=4)
    assert len(batch.sequences) == 41
    assert batch.end_indices[0] == 7
    assert batch.end_indices == sorted(batch.end_indices)
    assert batch.targets[0] == float(rows[7]["target"])
