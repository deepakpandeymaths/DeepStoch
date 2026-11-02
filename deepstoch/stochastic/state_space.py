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


class LocalLinearTrendModel:
    """Two-state Kalman model with latent level and latent trend."""

    def __init__(self, level_variance: float = 1.0, trend_variance: float = 0.1, observation_variance: float = 1.0) -> None:
        if min(level_variance, trend_variance, observation_variance) <= 0:
            raise ValueError("all variances must be positive")
        self.level_variance = level_variance
        self.trend_variance = trend_variance
        self.observation_variance = observation_variance
        self.level: float | None = None
        self.trend: float | None = None
        self.covariance: list[list[float]] | None = None

    def fit(self, observations: list[float]) -> "LocalLinearTrendModel":
        if not observations:
            raise ValueError("observations must be non-empty")
        level = float(observations[0])
        trend = float(observations[1] - observations[0]) if len(observations) > 1 else 0.0
        covariance = [[self.observation_variance, 0.0], [0.0, self.observation_variance]]
        for observation in observations[1:]:
            level += trend
            covariance = [[covariance[0][0] + covariance[0][1] + covariance[1][0] + covariance[1][1] + self.level_variance, covariance[0][1] + covariance[1][1]], [covariance[1][0] + covariance[1][1], covariance[1][1] + self.trend_variance]]
            innovation_variance = covariance[0][0] + self.observation_variance
            gain_level = covariance[0][0] / innovation_variance
            gain_trend = covariance[1][0] / innovation_variance
            innovation = observation - level
            level += gain_level * innovation
            trend += gain_trend * innovation
            covariance = [[(1 - gain_level) * covariance[0][0], (1 - gain_level) * covariance[0][1]], [covariance[1][0] - gain_trend * covariance[0][0], covariance[1][1] - gain_trend * covariance[0][1]]]
        self.level, self.trend, self.covariance = level, trend, covariance
        return self

    def forecast(self, horizon: int, confidence_z: float = 1.645) -> StateSpaceForecast:
        if self.level is None or self.trend is None or self.covariance is None:
            raise ValueError("model must be fitted before forecasting")
        if horizon < 1 or confidence_z <= 0:
            raise ValueError("horizon must be positive and confidence_z must be positive")
        means, lower, upper = [], [], []
        for step in range(1, horizon + 1):
            mean = self.level + step * self.trend
            variance = self.covariance[0][0] + 2 * step * self.covariance[0][1] + step * step * self.covariance[1][1] + self.observation_variance
            spread = confidence_z * math.sqrt(max(variance, 0.0))
            means.append(mean)
            lower.append(mean - spread)
            upper.append(mean + spread)
        return StateSpaceForecast(means, lower, upper, self.level, self.covariance[0][0])
