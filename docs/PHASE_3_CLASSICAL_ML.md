# Phase 3 — Classical machine-learning forecasting

Phase 3 adds the first supervised machine-learning experiment to DeepStoch.
The feature builder creates lagged targets, a rolling mean, cyclical hour
features, temperature, and holiday indicators. Every target lag is taken from
earlier observations only; the current target is never used as an input.

The first model is an ordinary least-squares linear regression implemented with
the Python standard library. This transparent implementation makes the
experiment reproducible before optional scientific dependencies are introduced.

Run the experiment:

```powershell
python -m deepstoch ml data/demo_energy_demand.csv --test-size 8
python -m deepstoch ml data/demo_energy_demand.csv --test-size 8 --json
```

The chronological holdout remains untouched during fitting. The resulting MAE,
RMSE, and MAPE can be compared with the Phase 2 baselines. Later Phase 3 work
will add scikit-learn tree models and a shared experiment registry.
