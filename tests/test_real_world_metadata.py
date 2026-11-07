import json
from pathlib import Path

from scripts.convert_uci_appliances import convert


def test_uci_metadata_is_traceable() -> None:
    path = Path(__file__).parents[1] / "data" / "metadata" / "uci_electricity_load.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["repository_id"] == 321
    assert metadata["license"] == "CC BY 4.0"
    assert metadata["doi"] == "10.24432/C58C86"


def test_appliances_metadata_is_traceable() -> None:
    path = Path(__file__).parents[1] / "data" / "metadata" / "uci_appliances_energy.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["repository_id"] == 374
    assert metadata["target_column"] == "Appliances"
    assert metadata["license"] == "CC BY 4.0"


def test_appliances_converter_maps_target_and_covariates(tmp_path: Path) -> None:
    source = tmp_path / "energydata_complete.csv"
    output = tmp_path / "canonical.csv"
    source.write_text("date,Appliances,T1,RH_1,random\n2026-01-01 00:00:00,60,20,40,0.1\n", encoding="utf-8")
    assert convert(source, output) == 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp,target,T1,RH_1"
    assert lines[1] == "2026-01-01T00:00:00,60,20,40"
