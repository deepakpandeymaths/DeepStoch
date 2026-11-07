# DeepStoch Project Roadmap

## Vision

DeepStoch is a long-term software platform combining deep neural networks, classical machine learning, stochastic processes, uncertainty estimation, simulation, model comparison, and decision support.

## Dissertation objective

To design and implement a modular software platform that combines neural forecasting with stochastic-process simulation and uncertainty estimation for data-driven decision support.

## First complete product

The first release will accept a time-series dataset, validate and preprocess it, train baseline and neural models, produce probabilistic forecasts, estimate uncertainty, simulate possible future trajectories, detect unusual observations, and present the results through a dashboard.

## Current implementation status

| Phase | Status | Evidence |
|---|---|---|
| 0. Foundation | Completed | Python package, CLI, tests, SOP, report, GPL-3.0 licence |
| 1. Data foundation | Completed | CSV validation, chronological splitting, feature construction |
| 2. Baselines | Completed | Statistical baselines and common metrics |
| 3. Classical ML | Completed | Linear regression, Random Forest, leakage-safe features |
| 4. Neural sequence setup | Completed | Sequence builder and optional neural model foundation |
| 5. Uncertainty | Completed | Conformal intervals, neural predictive intervals, calibration reports |
| 6. Simulation | Completed | Bootstrap trajectories and threshold probabilities |
| 7. Dashboard | Completed | Browser dashboard with model and robustness evidence |
| 8. Hybrid research | In progress | State-space, ML-state hybrid, rolling evaluation, neural state-space, real-world neural folds |
| 9. Dissertation release | Pending | Real-world studies, final tables, user manual, and release archive |

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

### Phase 2 - Baselines and classical ML

Statistical and classical machine-learning models with common metrics, feature engineering, and experiment comparison.

### Phase 3 - Deep forecasting foundation

Sequence windows, optional PyTorch training, neural checkpoints, inference, and probabilistic output heads.

### Phase 4 - Probabilistic output and neural uncertainty

Quantile loss, distributional prediction, prediction intervals, coverage, and calibration.

### Phase 5 - Uncertainty engine

Conformal intervals, neural predictive scales, rolling calibration, deep ensembles, confidence warnings, and out-of-distribution analysis.

### Phase 6 - Stochastic simulation

Reusable process simulators and Monte Carlo scenario generation.

### Phase 7 - Dashboard

Browser workflow, model comparison, rolling evidence, uncertainty summaries, simulation results, roadmap status, and reports.

### Phase 8 - Hybrid research

Local-level and local-trend state-space models, ML-plus-residual-state correction, rolling-origin evaluation, neural state-space training, checkpoint inference, and neural-SDE prototypes.

### Phase 9 - Dissertation release

Real-world datasets, frozen implementation, reproducible demonstrations, unified model tables, calibration studies, user manual, and dissertation evidence archive.

## Immediate next work

1. Complete acquisition and conversion of the registered UCI electricity dataset.
2. Run validation and model comparison on selected client series.
3. Run rolling-origin experiments on real-world data, including dataset-specific neural retraining.
4. Add real-world comparison results to the dashboard and project report.
5. Repeat across additional domains before making general research claims.
6. Add ablations, significance analysis, and final dissertation artefacts.

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

The dissertation release must support a clean-environment workflow from data upload to validated forecast, uncertainty visualization, stochastic simulation, model comparison, and report export. Every result must be linked to a dataset, licence/provenance record, configuration, code revision, and evaluation record.

