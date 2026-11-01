"""Small dependency-free ordinary least-squares regressor for Phase 3."""

from __future__ import annotations


class LinearRegression:
    """Ordinary least squares with an intercept, implemented for transparency."""

    def __init__(self) -> None:
        self.coefficients: list[float] = []

    def fit(self, features: list[list[float]], targets: list[float]) -> "LinearRegression":
        if not features or len(features) != len(targets):
            raise ValueError("features and targets must have the same non-zero length")
        width = len(features[0])
        if width == 0 or any(len(row) != width for row in features):
            raise ValueError("feature rows must have a consistent non-zero width")
        design = [[1.0, *row] for row in features]
        normal = [[sum(row[i] * row[j] for row in design) for j in range(width + 1)] for i in range(width + 1)]
        # A tiny ridge term keeps redundant or constant research features from
        # making the normal equations singular while leaving the fit effectively
        # equivalent to ordinary least squares at this baseline stage.
        for index in range(1, width + 1):
            normal[index][index] += 1e-8
        vector = [sum(row[i] * target for row, target in zip(design, targets)) for i in range(width + 1)]
        self.coefficients = _solve(normal, vector)
        return self

    def predict(self, features: list[list[float]]) -> list[float]:
        if not self.coefficients:
            raise ValueError("model must be fitted before prediction")
        width = len(self.coefficients) - 1
        if any(len(row) != width for row in features):
            raise ValueError("prediction feature width does not match fitted model")
        return [self.coefficients[0] + sum(coef * value for coef, value in zip(self.coefficients[1:], row)) for row in features]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small linear system with pivoted Gaussian elimination."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("cannot fit linear regression: singular feature matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]
