# DeepStoch Standard Operating Procedure

## Purpose

This SOP controls the design, implementation, testing, evaluation, documentation, and release of the DeepStoch dissertation software project.

## Product definition

DeepStoch is an uncertainty-aware deep-learning and stochastic-modelling platform. It will answer:

1. What is likely to happen?
2. How uncertain is the prediction?
3. What other future trajectories are plausible?
4. What action should be considered when uncertainty is high?

## Tools

| Area | Tools |
|---|---|
| Language | Python |
| Numerical computing | NumPy, pandas, SciPy |
| Classical ML | scikit-learn, statsmodels |
| Deep learning | PyTorch |
| API | FastAPI, Pydantic |
| Frontend | React, TypeScript, Plotly or ECharts |
| Tracking | MLflow |
| Storage | SQLite initially, PostgreSQL later |
| Quality | pytest, Ruff, mypy |
| Packaging | Git, Docker, Docker Compose |

## Operating principles

- Start with a simple baseline.
- Separate training, validation, and test data chronologically.
- Record data, code, configuration, seed, model, metrics, and artifacts.
- Report uncertainty with empirical coverage and calibration evidence.
- Keep stochastic simulators independently testable.
- Do not commit secrets or private data.
- Treat predictions and simulations as estimates, not guarantees.
- A feature is incomplete until tests and documentation exist.

## Architecture

```text
Data -> Validation -> Preprocessing -> Models -> Uncertainty -> Evaluation -> Dashboard
                                      |                  |
                                      +-> Simulation -----+
```

## Model methods

The implementation order is: naive and statistical baselines; classical ML; MLP, CNN, RNN, LSTM, and GRU; quantile and distributional forecasting; Monte Carlo dropout and deep ensembles; stochastic processes; neural state-space models; neural SDEs; and optional advanced modules.

## Data procedure

For every dataset record its source, access date, license, fields, units, timestamp meaning, target variable, missing-value meaning, and known limitations. Validate file type, schema, data types, timestamps, duplicates, missing values, irregular intervals, target availability, and leakage. Fit preprocessing only on training data.

Use chronological splits:

```text
earlier period -> training
middle period  -> validation
latest period  -> test
```

## Training procedure

Every model run must load a versioned dataset, use a declared configuration, record random seeds, train only on training data, monitor validation data, save the best checkpoint, log metrics and artifacts, evaluate once on untouched test data, and record limitations.

## Probabilistic forecasting

For a quantile level $\tau$, use pinball loss. For distributional models, record the predicted distribution parameters and negative log-likelihood. Every interval must state nominal coverage, such as 80\% or 95\%, and must be checked against empirical held-out coverage.

## Uncertainty procedure

Distinguish aleatoric uncertainty, epistemic uncertainty, and distribution-shift uncertainty. Add methods in this order: Monte Carlo dropout, deep ensembles, bootstrap, conformal prediction, and out-of-distribution detection. A confidence score must document its inputs and limitations.

## Stochastic simulation procedure

Each simulator must declare process type, initial state, parameters, time step, horizon, number of trajectories, seed, boundary conditions, and stopping conditions. It must provide parameter validation, a sanity-check case, sample-path plots, summary statistics, and exportable trajectories.

## Evaluation

Point forecasts use MAE, RMSE, mean error, and horizon-wise error. Probabilistic forecasts use negative log-likelihood, pinball loss, interval coverage, interval width, calibration error, and CRPS where implemented. Anomaly detection uses precision, recall, F1, false-alarm rate, and detection delay.

## Testing

Unit tests cover validators, preprocessors, windows, metrics, probability calculations, simulators, tensor shapes, models, and API schemas. Integration tests cover upload-to-validation, training-to-checkpoint, checkpoint-to-inference, forecast-to-uncertainty, and dashboard workflows. Regression tests use a fixed demonstration dataset. Security tests cover file limits, path traversal, serialization, secret exposure, and future sandboxed execution.

## Experiment tracking

Every run receives an identifier and records dataset ID, preprocessing configuration, model type, hyperparameters, random seed, code revision, hardware, metrics, plots, artifacts, and limitations. MLflow is the standard tracking layer.

## Release procedure

1. Freeze the release scope.
2. Run the full test suite.
3. Run the clean-environment demonstration.
4. Verify documentation and dependency versions.
5. Review metrics and limitations.
6. Create a release tag.
7. Archive the release report and dissertation evidence.

## Risks and controls

| Risk | Control |
|---|---|
| Scope expansion | Freeze v0.1 before advanced models |
| Data leakage | Chronological split and leakage checks |
| Misleading uncertainty | Coverage and calibration reports |
| Reproducibility failure | Track code, data, configuration, seed, and environment |
| Limited compute | CPU-compatible first release |
| Unsafe automation | Sandboxing, limits, and human review |

## Definition of done

A feature is complete only when implementation, tests, manual verification, documentation, reproducible configuration, limitations, and acceptance criteria are all recorded.

