# Phase 5 — Uncertainty quantification

Phase 5 adds split-conformal prediction intervals to DeepStoch. The earliest
chronological observations train the point model, the next block calibrates
absolute residuals, and the final block is used only for evaluation. The
calibration residual quantile becomes a constant-radius prediction interval.

Run the experiment:

```powershell
python -m deepstoch uncertainty data/demo_energy_demand.csv --coverage 0.90
python -m deepstoch uncertainty data/demo_energy_demand.csv --coverage 0.90 --json
python -m deepstoch uncertainty data/demo_energy_demand.csv --coverages 0.80,0.90,0.95 --json
```

The report includes nominal coverage, calibration size, interval radius,
empirical test coverage, and mean interval width. This is a first calibrated
uncertainty layer; later work will add conditional intervals, probabilistic
neural forecasts, and calibration curves. The multi-level command provides the
first coverage-versus-width table for this analysis.
