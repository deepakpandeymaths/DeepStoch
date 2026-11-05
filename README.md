# DeepStoch

**DeepStoch: An Uncertainty-Aware Deep Learning and Stochastic Modelling Platform**

DeepStoch combines neural networks, classical machine learning, stochastic processes, simulation, and uncertainty estimation to produce probabilistic forecasts and support decisions under uncertainty. The central question is not only “what will happen next?” but also “how uncertain is that forecast, and is the uncertainty calibrated?”

## Current status

Phase 8 - Hybrid research foundation. Version `0.1.0.dev0`.

The repository is an active dissertation software prototype. The small demo dataset is used for software verification; it is not sufficient for final scientific claims.

The first product target is a probabilistic time-series forecasting platform with:

- CSV time-series validation;
- statistical and classical machine-learning baselines;
- LSTM or GRU forecasting;
- prediction intervals;
- uncertainty estimation;
- stochastic simulation;
- model comparison and reports;
- rolling-origin evaluation;
- browser dashboard;
- reproducible synthetic benchmark generation;
- registered real-world dataset provenance;
- GPL-3.0-or-later licensing.

Read [DEEPSTOCH_ROADMAP.md](DEEPSTOCH_ROADMAP.md) for the long-term plan, [the complete project report](docs/DEEPSTOCH_PROJECT_REPORT.tex), and [the editable SOP](deepstoch_sop/main.tex).

## Quick start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[core,dev]"
python -m deepstoch health
python -m pytest
```

The health check intentionally uses no third-party package so the environment can be diagnosed before the complete ML stack is installed.

## Experiment commands

```powershell
python -m deepstoch validate-data data/demo_energy_demand.csv
python -m deepstoch baseline data/demo_energy_demand.csv --test-size 8
python -m deepstoch ml data/demo_energy_demand.csv --model all --test-size 8
python -m deepstoch uncertainty data/demo_energy_demand.csv --coverages 0.80,0.90,0.95
python -m deepstoch simulate data/demo_energy_demand.csv --horizon 24 --paths 1000
python -m deepstoch hybrid data/demo_energy_demand.csv --model all --test-size 8
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv --minimum-train 24 --test-size 4 --step 4
python -m deepstoch compare data/demo_energy_demand.csv --test-size 8 --json
```

To open the dashboard:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/dashboard/` in a browser.

## Larger and real-world data

Generate a deterministic synthetic stress-test dataset:

```powershell
python scripts/generate_benchmark_dataset.py data/generated/synthetic_energy_720.csv --rows 720 --seed 42
python -m deepstoch rolling-hybrid data/generated/synthetic_energy_720.csv --minimum-train 480 --test-size 24 --step 24
```

This data is explicitly synthetic and must not be presented as real-world evidence. The first registered real-world source is [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112); its provenance, CC BY 4.0 licence, DOI, and conversion instructions are in [REAL_WORLD_DATA.md](docs/REAL_WORLD_DATA.md).

## Repository layout

```text
deepstoch/       Python package, CLI, models, data, experiments, and evaluation
dashboard/       Static research dashboard prototype
data/            Demo, metadata, generated-data, and raw-data locations
scripts/         Dataset-generation and real-world conversion utilities
tests/           Automated tests
docs/            Phase documents, project report, and data notes
configs/         Versioned configuration files
deepstoch_sop/   Editable LaTeX SOP project
artifacts/       Local model checkpoints; ignored by Git
LICENSE          GNU GPL v3.0 or later
pyproject.toml   Package and optional dependencies
```

## License

DeepStoch is released under the GNU General Public License v3.0 or later.
See [LICENSE](LICENSE) for the complete terms.
