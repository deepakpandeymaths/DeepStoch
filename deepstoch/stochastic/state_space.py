"""Local-level Gaussian state-space filtering."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StateSpaceForecast:
    mean: list[float]
    lower: list[float]
    upper: list[float]
    state_mean: float
    state_variance: float


class LocalLevelModel:
    """A transparent latent-level model: x_t=x_{t-1}+w_t, y_t=x_t+v_t."""

    def __init__(self, process_variance: float = 1.0, observation_variance: float = 1.0) -> None:
        if process_variance <= 0 or observation_variance <= 0:
            raise ValueError("process and observation variances must be positive")
        self.process_variance = process_variance
        self.observation_variance = observation_variance
        self.state_mean: float | None = None
        self.state_variance: float | None = None

    def fit(self, observations: list[float]) -> "LocalLevelModel":
        if not observations:
            raise ValueError("observations must be non-empty")
        mean = float(observations[0])
        variance = self.observation_variance
        for observation in observations[1:]:
            prior_variance = variance + self.process_variance
            gain = prior_variance / (prior_variance + self.observation_variance)
            mean = mean + gain * (observation - mean)
            variance = (1 - gain) * prior_variance
        self.state_mean = mean
        self.state_variance = variance
        return self

    def forecast(self, horizon: int, confidence_z: float = 1.645) -> StateSpaceForecast:
        if self.state_mean is None or self.state_variance is None:
            raise ValueError("model must be fitted before forecasting")
        if horizon < 1 or confidence_z <= 0:
            raise ValueError("horizon must be positive and confidence_z must be positive")
        means, lower, upper = [], [], []
        for step in range(1, horizon + 1):
            variance = self.state_variance + step * self.process_variance + self.observation_variance
            spread = confidence_z * math.sqrt(variance)
            means.append(self.state_mean)
            lower.append(self.state_mean - spread)
            upper.append(self.state_mean + spread)
        return StateSpaceForecast(means, lower, upper, self.state_mean, self.state_variance)
