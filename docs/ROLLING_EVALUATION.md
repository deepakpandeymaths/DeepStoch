# Rolling-origin evaluation

DeepStoch now supports repeated chronological evaluation for the ML-plus-state
hybrid model. Each fold trains only on observations available before its test
block. The report aggregates mean and standard deviation of RMSE, mean interval
coverage, and mean interval width.

Run it with:

```powershell
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv --minimum-train 24 --test-size 4 --step 4
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv --json
```

This is the evaluation structure required before making dissertation-scale
claims. The demo remains too small for final conclusions, but the same command
can be applied to larger real-world datasets without changing the methodology.
