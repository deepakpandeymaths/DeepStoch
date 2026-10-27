# DeepStoch

**DeepStoch: An Uncertainty-Aware Deep Learning and Stochastic Modelling Platform**

DeepStoch will combine neural networks, classical machine learning, stochastic processes, simulation, and uncertainty estimation to produce probabilistic forecasts and support decisions under uncertainty.

## Current status

Phase 0 - Foundation. Version `0.1.0.dev0`.

The first product target is a probabilistic time-series forecasting platform with:

- CSV time-series validation;
- statistical and classical machine-learning baselines;
- LSTM or GRU forecasting;
- prediction intervals;
- uncertainty estimation;
- stochastic simulation;
- model comparison and reports.

Read [DEEPSTOCH_ROADMAP.md](DEEPSTOCH_ROADMAP.md) for the long-term plan and [DEEPSTOCH_SOP.md](DEEPSTOCH_SOP.md) for the operating procedure.

## Quick start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[core,dev]"
python -m deepstoch health
python -m pytest
```

The health check intentionally uses no third-party package so the environment can be diagnosed before the complete ML stack is installed.

## Repository layout

```text
deepstoch/       Python package and CLI
models/          ML and deep-learning models
stochastic/      Stochastic-process implementations
uncertainty/     Uncertainty and calibration methods
evaluation/      Metrics and model comparison
data/            Dataset metadata and local data locations
experiments/     Experiment configurations and run records
tests/           Automated tests
docs/            Technical design documents
configs/         Versioned configuration files
deepstoch_sop/   Editable LaTeX SOP project
```

