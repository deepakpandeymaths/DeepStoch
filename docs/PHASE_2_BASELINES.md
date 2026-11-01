# Phase 2 — Baseline forecasting layer

Phase 2 establishes the first measurable modelling capability in DeepStoch. It
uses a chronological holdout and compares four transparent reference models:

- last observed value;
- historical mean;
- three-observation moving average; and
- 24-step seasonal naive forecast, with a safe fallback for short histories.

The evaluation reports MAE, RMSE, and MAPE. Models are ranked by RMSE, but all
three metrics are retained for later comparison with probabilistic and neural
models. No future observations are used to construct the history passed to a
forecaster.

Run the experiment from the repository root:

```powershell
python -m deepstoch baseline data/demo_energy_demand.csv --test-size 8
python -m deepstoch baseline data/demo_energy_demand.csv --test-size 8 --json
```

This is a reference layer, not the final research contribution. It gives the
dissertation a reproducible performance floor before feature engineering,
stochastic state-space models, uncertainty estimation, and deep learning are
introduced.
