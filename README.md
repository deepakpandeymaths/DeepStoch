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

## 1. Project meaning

The name DeepStoch combines **deep learning** and **stochastic modelling**.
Deep learning represents neural networks, learned temporal features, latent
representations, and sequence models. Stochastic modelling represents
randomness, probability distributions, latent states, simulation, and
uncertainty.

DeepStoch is designed to answer two questions:

1. What is the most likely future value?
2. How uncertain is that prediction, and is the uncertainty reliable?

An ordinary forecasting system may output:

```text
Tomorrow's value = 620
```

DeepStoch aims to output:

```text
Expected value          = 620
90% prediction interval = [580, 665]
Empirical coverage      = measured on held-out future observations
Simulation output       = many plausible future trajectories
Calibration status      = reported rather than assumed
```

The interval is not a guarantee. It is a statistical statement that must be
tested on observations not used for fitting or calibration.

## 2. Research problem and questions

Time-series data can contain noise, seasonality, trends, temporal dependence,
regime changes, missing variables, and distribution shift. A single point
prediction hides these limitations. DeepStoch makes them measurable through
point forecasts, intervals, simulated trajectories, calibration, and repeated
temporal evaluation.

The central research question is:

> How can deep-learning forecasts and stochastic-process models be combined to provide accurate, calibrated, interpretable, and reproducible forecasts for time-series data?

Supporting questions are:

1. How do statistical, classical ML, deep-learning, stochastic, and hybrid models compare under chronological evaluation?
2. Which uncertainty method gives the best coverage–width trade-off?
3. Does stochastic residual modelling improve a learned forecast?
4. Does a neural state-space model improve accuracy over an interpretable ML–state hybrid?
5. How stable are accuracy and calibration across rolling temporal folds?
6. How much data is required before empirical coverage becomes stable?
7. How does uncertainty change during unusual regimes or distribution shift?
8. Can simulated future paths support decisions better than a single forecast?

## 3. End-to-end software workflow

```text
Input time-series dataset
        ↓
Validate schema, timestamps, targets, and ordering
        ↓
Create chronological train/validation/calibration/test splits
        ↓
Generate lag, rolling, calendar, and covariate features
        ↓
Create chronological neural sequence windows
        ↓
Train baselines, ML, neural, stochastic, and hybrid models
        ↓
Generate point forecasts and predictive uncertainty
        ↓
Calibrate prediction intervals
        ↓
Generate Monte Carlo future trajectories
        ↓
Evaluate accuracy, coverage, width, and robustness
        ↓
Compare models and export JSON/CSV/dashboard reports
```

## 4. Complete implementation inventory

| Area | Implementation |
|---|---|
| Data loading | Dependency-light CSV loading with headers and row dictionaries |
| Validation | Required columns, timestamps, numeric targets, duplicates, ordering, and missing values |
| Splitting | Chronological train/validation/test splitting without shuffling |
| Features | Lagged targets, rolling means, cyclical hour features, temperature, and holiday indicators |
| Baselines | Last value, historical mean, moving average, and seasonal naive |
| Classical ML | Ordinary least-squares linear regression and Random Forest adapter |
| Neural sequences | Past-only chronological feature windows |
| Neural state-space | GRU encoder, latent transition, probabilistic mean, and predictive scale |
| Training | Training NLL, validation NLL, early stopping, and checkpoints |
| Inference | Neural checkpoint loading and chronological test evaluation |
| Uncertainty | Split-conformal intervals and neural predictive intervals |
| Simulation | Residual-bootstrap Monte Carlo trajectories and threshold probabilities |
| State-space | Local-level and local-linear-trend models |
| Hybrid modelling | Linear ML forecast plus latent residual-state correction |
| Evaluation | MAE, RMSE, MAPE, coverage, interval width, and rolling-origin metrics |
| Comparison | Unified chronological comparison across model families |
| Dashboard | Model, uncertainty, simulation, rolling, and roadmap views |
| Data registration | Synthetic benchmark generator and UCI electricity converter |
| Documentation | SOP, complete LaTeX report, phase documents, and roadmap |
| Licence | GNU GPL v3.0 or later |

## 5. Phase details

### Phase 0 — Foundation

Completed: Python package structure, CLI, health check, configuration
foundation, tests, roadmap, LaTeX SOP, complete LaTeX report, GitHub
repository, and GPL-3.0-or-later licence.

### Phase 1 — Data foundation

Completed: CSV loading, required-column checks, timestamp parsing,
missing-target checks, non-numeric-target checks, duplicate detection,
unsorted-data warnings, chronological splitting, and structured reports.

```powershell
python -m deepstoch validate-data data/demo_energy_demand.csv
```

### Phase 2 — Baseline forecasting

Completed models: last observed value, historical mean, moving average, and
seasonal naive. These establish a transparent performance floor before complex
models are introduced.

```powershell
python -m deepstoch baseline data/demo_energy_demand.csv --test-size 8
```

### Phase 3 — Classical machine learning

Implemented: lag-1/2/3 features, rolling mean, cyclical hour encoding,
temperature, holiday, linear regression, Random Forest, and leakage-safe
chronological evaluation.

```powershell
python -m deepstoch ml data/demo_energy_demand.csv --model all --test-size 8
```

### Phase 4 — Neural sequence foundation

Implemented: past-only sequence windows, optional PyTorch models, GRU neural
state-space encoder, latent state transition, probabilistic mean, and
predictive scale.

### Phase 5 — Uncertainty quantification

Implemented: split-conformal calibration, residual quantiles, nominal and
empirical coverage, mean interval width, multi-level calibration, and neural
Gaussian predictive intervals.

```powershell
python -m deepstoch uncertainty data/demo_energy_demand.csv `
  --coverages 0.80,0.90,0.95 --json
```

### Phase 6 — Stochastic simulation

Implemented: historical innovation sampling, reproducible seeds, Monte Carlo
paths, trajectory quantiles, and threshold-crossing probabilities.

```powershell
python -m deepstoch simulate data/demo_energy_demand.csv `
  --horizon 24 --paths 1000 --threshold 700
```

### Phase 7 — Dashboard

Implemented: browser dashboard, model table, uncertainty summary, simulation
summary, rolling-origin evidence, roadmap status, and responsive layout.

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/dashboard/`.

### Phase 8 — Hybrid research

Implemented: local-level state-space, local-linear-trend state-space,
ML-plus-residual-state correction, rolling-origin evaluation, normalized neural
state-space training, validation NLL, early stopping, checkpoint saving,
checkpoint inference, and neural accuracy/coverage evaluation.

```powershell
python -m deepstoch hybrid data/demo_energy_demand.csv --model all --test-size 8
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv --minimum-train 24 --test-size 4 --step 4
```

### Phase 9 — Dissertation release

Planned: real-world multi-domain experiments, ablation studies, statistical
analysis, frozen configurations, final dashboard demonstration, dissertation
tables and figures, user manual, reproducibility archive, and final release.

## 6. Neural state-space model

The current architecture is:

```text
Feature sequence
      ↓
GRU encoder
      ↓
Latent state
      ↓
Neural transition network
      ↓
Mean head + predictive-scale head
      ↓
Probabilistic forecast
```

Training uses Gaussian negative log-likelihood. Features and targets are
standardized using training data only. Checkpoints store the model state,
input size, epoch, validation loss, feature statistics, and target statistics.

Install and train:

```powershell
python -m pip install -e ".[deep-learning]"
python -m deepstoch neural-state data/demo_energy_demand.csv `
  --sequence-length 6 --epochs 50 `
  --checkpoint artifacts/neural_state_space.pt
```

Evaluate the checkpoint:

```powershell
python -m deepstoch neural-evaluate data/demo_energy_demand.csv `
  --checkpoint artifacts/neural_state_space.pt --test-size 8 --json
```

The neural model is not automatically considered superior. Its accuracy,
interval quality, stability, and computational cost must be compared with the
classical and hybrid baselines.

## 7. Unified model comparison

```powershell
python -m deepstoch compare data/demo_energy_demand.csv `
  --test-size 8 --json
```

The report currently evaluates seasonal naive, last value, moving average,
historical mean, linear regression, Random Forest, local-level state-space,
local-linear-trend state-space, linear ML plus residual state, and neural
state-space when a checkpoint is available.

Reported fields include MAE, RMSE, MAPE, empirical coverage, mean interval
width, and model status. The demo result is a software checkpoint, not a general
scientific ranking, because the fixture is very small.

## 8. Rolling-origin evaluation

Rolling-origin evaluation repeatedly trains on the past and evaluates on the
next chronological block:

```text
Fold 1: early history  → next test block
Fold 2: larger history → next test block
Fold 3: larger history → next test block
...
```

```powershell
python -m deepstoch rolling-hybrid data/demo_energy_demand.csv `
  --minimum-train 24 --test-size 4 --step 4 --json
```

For the probabilistic neural state-space model, use `rolling-neural`:

```powershell
python -m deepstoch rolling-neural data/demo_energy_demand.csv `
  --sequence-length 6 --minimum-train 120 --test-size 24 --step 24 `
  --epochs 20 --patience 5 --json
```

Every neural fold is retrained from scratch. Feature and target normalization
statistics are calculated inside that fold's training prefix, so future test
observations cannot influence model fitting or scaling.

To test seed sensitivity, repeat the rolling experiment:

```powershell
python -m deepstoch rolling-neural-repeated data/real_world/uci_appliances.csv `
  --minimum-train 12000 --test-size 672 --step 672 `
  --epochs 6 --patience 3 --seeds 42,43,44 --json
```

The repeated report includes aggregate RMSE and coverage variation together
with the complete result for each seed.

The report includes fold count, mean RMSE, RMSE standard deviation, mean
coverage, mean interval width, and fold-level results. This is the required
structure for dissertation-scale temporal claims.

## 9. Mathematical foundations

Let $y_t$ be the target at time $t$ and $\mathbf{x}_t$ the observed covariates.
A point forecast is represented as:

```text
y_hat(t+h) = f_theta(y_1:t, x_1:t+h)
```

The stochastic formulation represents the future as a conditional distribution:

```text
Y_(t+1:t+H) ~ p_theta(Y_(t+1:t+H) | Y_1:t, X)
```

For an interval $[L_t,U_t]$, empirical coverage is the proportion of future
observations that lie inside their corresponding intervals. Mean interval width
is the average of $U_t-L_t$. A useful uncertainty method should achieve
coverage close to its nominal level without making intervals unnecessarily
wide.

## 10. Evaluation metrics

Point metrics:

- **MAE**: average absolute error;
- **RMSE**: square root of average squared error;
- **MAPE**: percentage error with zero-valued actuals handled explicitly.

Uncertainty metrics:

- nominal coverage;
- empirical coverage;
- mean interval width;
- calibration across multiple nominal levels;
- coverage variation across rolling folds;
- future interval score.

Operational metrics planned for later experiments include training time,
inference time, memory use, checkpoint size, and reproducibility across random
seeds.

## 11. Data policy

### Demo data

`data/demo_energy_demand.csv` is a small fixture for automated tests and
examples.

### Synthetic benchmark

```powershell
python scripts/generate_benchmark_dataset.py `
  data/generated/synthetic_energy_720.csv --rows 720 --seed 42
```

The generated dataset contains deterministic daily/weekly patterns, trend,
temperature, holiday, and noise components. It is only for pipeline stress
testing and is not real-world evidence.

### UCI electricity source

The first registered real-world source is [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112). Its provenance, DOI, CC BY 4.0 licence, time resolution, client count, and source-format notes are in `data/metadata/uci_electricity_load.json` and [REAL_WORLD_DATA.md](docs/REAL_WORLD_DATA.md).

The raw archive is excluded from Git because of its size. After obtaining the
source file, convert one client:

```powershell
python scripts/convert_uci_electricity.py `
  path\to\LD2011_2014.txt `
  data\real_world\uci_client_0.csv `
  --client ACTUAL_CLIENT_COLUMN --divide-by-four
```

Then validate before modelling:

```powershell
python -m deepstoch validate-data data\real_world\uci_client_0.csv
```

Any dissertation use must preserve attribution and licence requirements.

## 12. Research methodology and reproducibility

Every experiment should record dataset name and checksum, source and licence,
selected target, timestamp convention, preprocessing, features, split
boundaries, model configuration, random seed, software version, Git commit,
metrics, and generated artefacts.

Future studies will include rolling-origin validation, feature ablations,
deterministic-versus-probabilistic comparisons, calibration curves,
coverage–width analysis, regime-shift analysis, multiple real-world domains,
repeated seeds, and statistical tests for paired temporal results.

## 13. Limitations and safety boundaries

DeepStoch is not currently a clinical diagnostic system, guaranteed prediction
engine, autonomous decision-maker, replacement for domain experts, finished
commercial product, or reliability-analysis-only system. It is research and
decision-support software.

The current demo results are limited by small data. High empirical coverage
alone does not imply useful uncertainty; intervals may be too wide. Real-world
conclusions require larger datasets, multiple folds, documented provenance, and
careful statistical evaluation.

## 14. Dissertation contribution

The intended contribution has four parts:

1. **Software contribution**: a modular open-source forecasting and uncertainty platform.
2. **Mathematical contribution**: a unified treatment of temporal forecasting, stochastic states, simulation, intervals, and calibration.
3. **Experimental contribution**: a fair comparison of baseline, ML, neural, stochastic, and hybrid models.
4. **Research contribution**: a future hybrid method or calibration strategy supported by reproducible experiments.

## 15. Future research

Planned extensions include quantile regression, distributional neural
forecasting, deep ensembles, Monte Carlo dropout, adaptive conformal
prediction, regime-switching state-space models, seasonal latent dynamics,
neural stochastic differential equations, Transformer forecasting, diffusion
models, out-of-distribution detection, anomaly detection, experiment history,
LaTeX table export, an optional API, and an optional LLM explanation interface.

## 16. Dissertation-release success criteria

The final release will be ready when a clean-environment user can:

1. install the repository;
2. validate a time-series dataset;
3. run baseline and ML models;
4. train or load a neural checkpoint;
5. generate uncertainty intervals;
6. simulate future trajectories;
7. run rolling-origin evaluation;
8. compare models using a documented protocol;
9. inspect results in the dashboard;
10. reproduce the dissertation tables and figures.

## 17. Documentation map

- [Project roadmap](DEEPSTOCH_ROADMAP.md)
- [Complete LaTeX project report](docs/DEEPSTOCH_PROJECT_REPORT.tex)
- [LaTeX SOP](deepstoch_sop/main.tex)
- [Phase 1 data foundation](docs/PHASE_1_DATA_FOUNDATION.md)
- [Phase 2 baselines](docs/PHASE_2_BASELINES.md)
- [Phase 3 classical ML](docs/PHASE_3_CLASSICAL_ML.md)
- [Phase 4 deep learning](docs/PHASE_4_DEEP_LEARNING.md)
- [Phase 5 uncertainty](docs/PHASE_5_UNCERTAINTY.md)
- [Phase 6 simulation](docs/PHASE_6_STOCHASTIC_SIMULATION.md)
- [Phase 7 dashboard](docs/PHASE_7_DASHBOARD.md)
- [Phase 8 hybrid research](docs/PHASE_8_HYBRID_RESEARCH.md)
- [Neural state-space foundation](docs/NEURAL_STATE_SPACE.md)
- [Neural evaluation](docs/NEURAL_EVALUATION.md)
- [Model comparison](docs/MODEL_COMPARISON.md)
- [Rolling evaluation](docs/ROLLING_EVALUATION.md)
- [Benchmark datasets](docs/BENCHMARK_DATASETS.md)
- [Real-world data registration](docs/REAL_WORLD_DATA.md)
