# Rolling-origin evaluation

DeepStoch now supports repeated chronological evaluation for the ML-plus-state
hybrid model. Each fold trains only on observations available before its test
block. The report aggregates mean and standard deviation of RMSE, mean interval
coverage, and mean interval width.

The same protocol is available for the probabilistic neural state-space model.
It retrains a fresh model at every origin, derives normalization statistics
from that fold's training prefix, and evaluates only the next unseen block.

Run it with:

```powershell
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv --minimum-train 24 --test-size 4 --step 4
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv --json
python -m deepstoch rolling-neural data/demo_energy_demand.csv --minimum-train 120 --test-size 24 --step 24 --epochs 20 --patience 5 --json
```

This is the evaluation structure required before making dissertation-scale
claims. The demo remains too small for final conclusions, but the same command
can be applied to larger real-world datasets without changing the methodology.

For the first real-world neural experiment, the following configuration was
used on the UCI Appliances Energy Prediction series:

```powershell
python -m deepstoch rolling-neural data/real_world/uci_appliances.csv `
  --sequence-length 6 --minimum-train 12000 --test-size 672 --step 672 `
  --epochs 8 --patience 3 --json
```

This produced 11 chronological folds with mean RMSE `91.1869`, RMSE standard
deviation `23.5358`, mean empirical coverage `94.7646%` at nominal 90%, and
mean interval width `331.2871`. These are initial research measurements, not
general performance claims; seed sensitivity, calibration analysis, and
additional datasets remain required.
