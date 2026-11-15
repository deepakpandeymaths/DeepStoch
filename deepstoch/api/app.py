"""Optional FastAPI adapter for the local DeepStoch experiment service."""

from __future__ import annotations

from pathlib import Path


def create_app():
    """Create the local API app; FastAPI remains an optional dependency."""
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError("API support requires pip install -e '.[api]'") from exc

    from deepstoch.api.service import dataset_summary, model_comparison

    app = FastAPI(title="DeepStoch Local Experiment API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "deepstoch-local-api"}

    @app.get("/dataset/summary")
    def dataset(path: str, time_column: str = "timestamp", target_column: str = "target") -> dict[str, object]:
        try:
            return dataset_summary(Path(path), time_column=time_column, target_column=target_column)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/models/compare")
    def compare(
        path: str,
        time_column: str = "timestamp",
        target_column: str = "target",
        test_size: int = Query(default=8, ge=1),
    ) -> dict[str, object]:
        try:
            return model_comparison(Path(path), time_column=time_column, target_column=target_column, test_size=test_size)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
