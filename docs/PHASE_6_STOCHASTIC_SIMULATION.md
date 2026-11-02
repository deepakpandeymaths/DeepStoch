# Phase 6 — Stochastic trajectory simulation

Phase 6 adds a reproducible Monte Carlo engine. Historical first differences
are treated as empirical innovations, sampled with a fixed seed, and accumulated
from the latest observation to create many plausible future trajectories.

Run a simulation:

```powershell
python -m deepstoch simulate data/demo_energy_demand.csv --horizon 24 --paths 1000 --threshold 700
python -m deepstoch simulate data/demo_energy_demand.csv --horizon 24 --paths 1000 --json
```

The output includes 10th, 50th, and 90th trajectory quantiles and, when a
threshold is supplied, the proportion of paths that cross it. This is a
transparent bootstrap process rather than a final domain-specific stochastic
model. Later work will add state-space dynamics, seasonal innovations, regime
switching, and neural stochastic processes.
