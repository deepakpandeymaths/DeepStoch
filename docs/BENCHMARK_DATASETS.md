# Benchmark datasets

The repository contains a small demo fixture for unit tests. For larger
pipeline stress tests, DeepStoch now includes a deterministic synthetic-data
generator:

```powershell
python scripts/generate_benchmark_dataset.py data/generated/synthetic_energy_720.csv --rows 720 --seed 42
python -m deepstoch validate-data data/generated/synthetic_energy_720.csv
python -m deepstoch rolling-hybrid data/generated/synthetic_energy_720.csv --minimum-train 480 --test-size 24 --step 24
```

The generated data is explicitly synthetic and must not be presented as
real-world dissertation evidence. It verifies that the pipeline scales beyond
the 48-row fixture. The next research data task is to add documented real-world
datasets with source attribution, usage licence, preprocessing notes, and fixed
dataset metadata.
