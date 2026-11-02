# Phase 8 — Hybrid state-space research foundation

Phase 8 begins the hybrid research layer with a local-level Gaussian
state-space model. The observed series is treated as a noisy measurement of a
latent level that evolves with process noise. The Kalman update estimates the
latent state, and the forecast interval grows with the future process horizon.

Run the experiment:

```powershell
python -m deepstoch hybrid data/demo_energy_demand.csv --test-size 8
python -m deepstoch hybrid data/demo_energy_demand.csv --test-size 8 --json
```

This is an interpretable stochastic baseline for future hybrid models. The next
research steps are to combine learned features with latent states, add seasonal
and regime-switching transitions, and prototype optional neural state-space and
neural stochastic differential-equation models.
