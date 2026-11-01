"""Optional scikit-learn tree model adapter."""

from __future__ import annotations


class RandomForestRegressor:
    """Small project adapter that keeps the optional dependency isolated."""

    def __init__(self, *, n_estimators: int = 200, random_state: int = 42) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor as SklearnForest
        except ImportError as exc:
            raise RuntimeError("Random Forest requires the optional 'core' dependencies") from exc
        self._model = SklearnForest(n_estimators=n_estimators, random_state=random_state, n_jobs=1)

    def fit(self, features: list[list[float]], targets: list[float]) -> "RandomForestRegressor":
        self._model.fit(features, targets)
        return self

    def predict(self, features: list[list[float]]) -> list[float]:
        return [float(value) for value in self._model.predict(features)]
