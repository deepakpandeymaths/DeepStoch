# Unified model comparison

DeepStoch now evaluates all available model families on one chronological
holdout and returns a single comparison report. The report includes point
accuracy for every model and interval coverage/width where the model produces
uncertainty.

Run it after training the neural checkpoint:

```powershell
python -m deepstoch compare data/demo_energy_demand.csv --test-size 8
python -m deepstoch compare data/demo_energy_demand.csv --test-size 8 --json
```

The output is the foundation for dissertation tables and dashboard integration.
The demo dataset remains a software verification fixture; final conclusions
require larger datasets and rolling-origin experiments.
