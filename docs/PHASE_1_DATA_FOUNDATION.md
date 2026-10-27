# Phase 1 - Data Foundation

## Objective

Provide the first reproducible data workflow for DeepStoch before implementing forecasting models.

## Implemented

- dependency-light CSV loader;
- required-column, timestamp, target, duplicate, and ordering validation;
- chronological train/validation/test splitting;
- structured dataset reports;
- `validate-data` CLI command;
- synthetic demonstration energy-demand dataset;
- automated normal and invalid-input tests.

## Commands

```powershell
python -m deepstoch validate-data data/demo_energy_demand.csv
python -m deepstoch validate-data data/demo_energy_demand.csv --json
python -m pytest
```

## Next stage

Phase 2 will add statistical and classical machine-learning baselines using this validated data contract.

