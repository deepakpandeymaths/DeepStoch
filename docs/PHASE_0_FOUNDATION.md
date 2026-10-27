# DeepStoch Phase 0 - Foundation

## Objective

Create an installable, testable, documented foundation before adding datasets or machine-learning models.

## Implemented

- Python package metadata;
- optional dependency groups for core ML, deep learning, API, tracking, and development;
- dependency-free health check;
- initial repository structure;
- default configuration;
- automated tests;
- README quick start;
- ignore rules for environments, data, models, and generated outputs.

## Acceptance criteria

- `python -m deepstoch health` reports `status: ok`;
- `python -m deepstoch health --json` returns valid JSON;
- `python -m pytest` passes;
- `python -m pip install -e .` succeeds;
- optional dependencies are not required for the health check;
- the repository contains no secrets or private data.

## Next stage

Phase 1 will implement CSV ingestion, dataset metadata, schema validation, timestamp checks, missing-value reports, chronological splitting, and a reproducible demonstration dataset.

