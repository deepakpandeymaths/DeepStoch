from pathlib import Path

from deepstoch.data.loader import load_csv


def test_benchmark_generator_is_available() -> None:
    script = Path(__file__).parents[1] / "scripts" / "generate_benchmark_dataset.py"
    source = script.read_text(encoding="utf-8")
    assert "synthetic benchmark" in source
    assert "temperature" in source
    assert "holiday" in source
