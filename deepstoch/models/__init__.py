"""Baseline and learned forecasting models."""

from deepstoch.models.baselines import (
    mean_forecast,
    moving_average_forecast,
    seasonal_naive_forecast,
    last_value_forecast,
)

__all__ = [
    "last_value_forecast",
    "mean_forecast",
    "moving_average_forecast",
    "seasonal_naive_forecast",
]
