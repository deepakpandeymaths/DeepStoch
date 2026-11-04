import json
from pathlib import Path


def test_uci_metadata_is_traceable() -> None:
    path = Path(__file__).parents[1] / "data" / "metadata" / "uci_electricity_load.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["repository_id"] == 321
    assert metadata["license"] == "CC BY 4.0"
    assert metadata["doi"] == "10.24432/C58C86"
