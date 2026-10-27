from pathlib import Path

from deepstoch import __version__
from deepstoch.core.health import find_repository_root, run_health_check


def test_health_report_is_ok() -> None:
    report = run_health_check(Path(__file__).parent)
    assert report.status == "ok"
    assert report.project == "DeepStoch"
    assert report.version == __version__
    assert Path(report.repository_root).joinpath("pyproject.toml").is_file()


def test_repository_root_contains_metadata() -> None:
    root = find_repository_root(Path(__file__).parent)
    assert root.joinpath("pyproject.toml").is_file()
    assert root.joinpath("README.md").is_file()

