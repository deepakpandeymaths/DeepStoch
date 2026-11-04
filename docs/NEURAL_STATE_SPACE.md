# Neural state-space foundation

DeepStoch now includes an optional neural state-space model. A GRU encoder maps
each chronological feature sequence to a latent state, a transition network
evolves that state, and probabilistic output heads produce a forecast mean and
positive predictive scale. Training uses a Gaussian negative
log-likelihood objective.

PyTorch is optional. Install it before training:

```powershell
python -m pip install -e ".[deep-learning]"
python -m deepstoch neural-state data/demo_energy_demand.csv --sequence-length 6 --epochs 50 --checkpoint artifacts/neural_state_space.pt
```

The training command creates a chronological train/validation split, tracks
validation negative log-likelihood, applies patience-based early stopping, and
saves the best checkpoint. Future work will add calibration, test-set inference,
and comparison with the ML-plus-state hybrid.

The first verified CPU training run used 39 sequences (31 training and 8
validation sequences) for 50 epochs. Its training NLL decreased from
296343.1875 to 3158.3936, with best validation NLL 3769.4744. The generated
checkpoint is intentionally ignored by Git because model binaries are local
experiment artefacts and should be managed separately from source code.
