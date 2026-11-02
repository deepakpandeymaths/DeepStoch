# Neural state-space foundation

DeepStoch now includes an optional neural state-space model. A GRU encoder maps
each chronological feature sequence to a latent state, a transition network
evolves that state, and probabilistic output heads produce a forecast mean and
positive predictive scale. Training uses a Gaussian negative
log-likelihood objective.

PyTorch is optional. Install it before training:

```powershell
python -m pip install -e ".[deep-learning]"
python -m deepstoch neural-state data/demo_energy_demand.csv --sequence-length 6 --epochs 50
```

The current environment does not contain PyTorch, so the CLI reports the
dependency requirement rather than claiming an untrained neural result. Future
work will add validation splits, checkpointing, calibration, and comparison
with the ML-plus-state hybrid.
