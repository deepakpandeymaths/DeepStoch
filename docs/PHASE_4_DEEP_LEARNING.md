# Phase 4 — Deep-learning forecasting foundation

Phase 4 introduces chronological sequence preparation and an optional PyTorch
LSTM forecaster. A sequence contains only feature rows observed before its
label, preserving the no-future-leakage rule used by the earlier phases.

The current implementation deliberately keeps PyTorch optional because it is a
large dependency. Install the project deep-learning extras before training:

```powershell
python -m pip install -e ".[deep-learning]"
python -m deepstoch neural data/demo_energy_demand.csv --sequence-length 6 --epochs 50
```

The CLI reports the number of sequences and training-loss reduction. The next
Phase 4 milestone will add a chronological validation split, standardized
features, checkpoint saving, and a direct comparison against the Phase 2 and
Phase 3 models.
