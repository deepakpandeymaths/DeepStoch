from pathlib import Path


DASHBOARD = Path(__file__).parents[1] / "dashboard"


def test_dashboard_assets_exist_and_have_entrypoint() -> None:
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert "DeepStoch" in html
    assert 'src="app.js"' in html
    assert (DASHBOARD / "styles.css").exists()
    assert (DASHBOARD / "app.js").exists()
