# DeepStoch Project Roadmap

## Vision

DeepStoch is a long-term software platform combining deep neural networks, classical machine learning, stochastic processes, uncertainty estimation, simulation, and decision support.

## Dissertation objective

To design and implement a modular software platform that combines neural forecasting with stochastic-process simulation and uncertainty estimation for data-driven decision support.

## First complete product

The first release will accept a time-series dataset, validate and preprocess it, train baseline and neural models, produce probabilistic forecasts, estimate uncertainty, simulate possible future trajectories, detect unusual observations, and present the results through a dashboard.

## Model progression

1. Naive, moving-average, seasonal, and statistical baselines.
2. Linear regression, random forest, gradient boosting, and quantile regression.
3. MLP, 1D CNN, RNN, LSTM, and GRU.
4. Quantile and distributional neural forecasting.
5. Monte Carlo dropout, deep ensembles, bootstrap, and conformal prediction.
6. Markov chains, Poisson processes, random walks, Brownian motion, and state-space models.
7. Neural state-space models and neural stochastic differential equations.
8. Optional Transformer, diffusion, reinforcement-learning, and LLM interface modules.

## Main phases

### Phase 0 - Foundation

Repository, Python package, configuration, health check, tests, documentation, and environment policy.

### Phase 1 - Data foundation

CSV loading, schema validation, timestamp checks, missing values, leakage checks, preprocessing, and chronological splitting.

### Phase 2 - Baselines

Statistical and classical machine-learning models with common metrics and experiment tracking.

### Phase 3 - Deep forecasting

PyTorch dataset windows, LSTM or GRU training, checkpoints, inference, and visualizations.

### Phase 4 - Probabilistic output

Quantile loss, distributional prediction, prediction intervals, coverage, and calibration.

### Phase 5 - Uncertainty engine

Monte Carlo dropout, deep ensembles, confidence warnings, and out-of-distribution analysis.

### Phase 6 - Stochastic simulation

Reusable process simulators and Monte Carlo scenario generation.

### Phase 7 - Dashboard

Browser workflow, run history, forecast charts, uncertainty plots, simulation results, and reports.

### Phase 8 - Hybrid research

Neural state-space and neural-SDE prototypes.

### Phase 9 - Dissertation release

Frozen implementation, reproducible demonstration, evaluation report, user manual, and dissertation evidence archive.

## Twenty-four-week plan

| Weeks | Work package | Evidence |
|---|---|---|
| 1-2 | Setup | Installable environment and architecture |
| 3-5 | Data foundation | Validated dataset and tests |
| 6-8 | Baselines | Model comparison report |
| 9-12 | Neural model | LSTM or GRU forecast |
| 13-15 | Probabilistic output | Intervals and coverage |
| 16-18 | Uncertainty | Calibration and confidence report |
| 19-21 | Simulation | Stochastic trajectories |
| 22-24 | Dashboard and release | End-to-end demonstration |

## Success criteria

The dissertation release must support a clean-environment workflow from data upload to validated forecast, uncertainty visualization, stochastic simulation, and report export. Every result must be linked to a dataset, configuration, code revision, and evaluation record.

