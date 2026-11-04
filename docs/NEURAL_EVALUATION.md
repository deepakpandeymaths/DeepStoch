# Neural state-space evaluation

The trained neural state-space checkpoint can now be evaluated on the final
chronological sequences:

```powershell
python -m deepstoch neural-evaluate data/demo_energy_demand.csv `
  --checkpoint artifacts/neural_state_space.pt `
  --sequence-length 6 `
  --test-size 8
```

The report includes MAE, RMSE, MAPE, nominal coverage, empirical coverage, and
mean prediction-interval width. JSON output is available with `--json` for
dashboard and dissertation tables.

The first normalized CPU evaluation produced MAE 67.1321, RMSE 73.3443, MAPE
10.19%, 100% empirical coverage at the 90% nominal level, and mean interval
width 267.8903. These numbers are only a software-pipeline checkpoint because
the demo dataset is very small; they are not final research conclusions.
