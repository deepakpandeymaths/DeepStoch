"""Command-line entry point for DeepStoch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepstoch.data.loader import DataLoadError, load_csv
from deepstoch.data.report import build_report
from deepstoch.data.split import chronological_split
from deepstoch.data.validation import validate_rows
from deepstoch.core.health import run_health_check
from deepstoch.experiments.baselines import evaluate_baselines
from deepstoch.experiments.ml import evaluate_linear_regression, evaluate_random_forest
from deepstoch.data.sequences import build_sequences_from_rows
from deepstoch.models.lstm import train_lstm
from deepstoch.experiments.uncertainty import evaluate_conformal_levels, evaluate_conformal_linear
from deepstoch.experiments.simulation import simulate_dataset
from deepstoch.experiments.hybrid import evaluate_local_level


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepstoch",
        description="DeepStoch uncertainty-aware modelling platform.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health", help="Check the local project environment.")
    health.add_argument("--json", action="store_true", help="Print JSON output.")
    validate = commands.add_parser("validate-data", help="Validate a CSV time-series dataset.")
    validate.add_argument("path", type=Path, help="Path to the CSV dataset.")
    validate.add_argument("--time-column", default="timestamp")
    validate.add_argument("--target-column", default="target")
    validate.add_argument("--json", action="store_true", help="Print the report as JSON.")
    baseline = commands.add_parser("baseline", help="Evaluate Phase 2 forecasting baselines.")
    baseline.add_argument("path", type=Path, help="Path to the CSV dataset.")
    baseline.add_argument("--target-column", default="target")
    baseline.add_argument("--test-size", type=int, default=8, help="Number of final observations used as holdout.")
    baseline.add_argument("--json", action="store_true", help="Print scores as JSON.")
    ml = commands.add_parser("ml", help="Evaluate Phase 3 classical ML models.")
    ml.add_argument("path", type=Path, help="Path to the CSV dataset.")
    ml.add_argument("--time-column", default="timestamp")
    ml.add_argument("--target-column", default="target")
    ml.add_argument("--test-size", type=int, default=8)
    ml.add_argument("--model", choices=["linear", "random-forest", "all"], default="all")
    ml.add_argument("--json", action="store_true", help="Print scores as JSON.")
    neural = commands.add_parser("neural", help="Train the optional Phase 4 LSTM model.")
    neural.add_argument("path", type=Path, help="Path to the CSV dataset.")
    neural.add_argument("--time-column", default="timestamp")
    neural.add_argument("--target-column", default="target")
    neural.add_argument("--sequence-length", type=int, default=6)
    neural.add_argument("--epochs", type=int, default=50)
    uncertainty = commands.add_parser("uncertainty", help="Evaluate Phase 5 conformal prediction intervals.")
    uncertainty.add_argument("path", type=Path, help="Path to the CSV dataset.")
    uncertainty.add_argument("--time-column", default="timestamp")
    uncertainty.add_argument("--target-column", default="target")
    uncertainty.add_argument("--calibration-size", type=int, default=8)
    uncertainty.add_argument("--test-size", type=int, default=8)
    uncertainty.add_argument("--coverage", type=float, default=0.9)
    uncertainty.add_argument("--coverages", help="Comma-separated levels, for example 0.8,0.9,0.95.")
    uncertainty.add_argument("--json", action="store_true")
    simulation = commands.add_parser("simulate", help="Generate Phase 6 stochastic future trajectories.")
    simulation.add_argument("path", type=Path, help="Path to the CSV dataset.")
    simulation.add_argument("--target-column", default="target")
    simulation.add_argument("--horizon", type=int, default=24)
    simulation.add_argument("--paths", type=int, default=1000)
    simulation.add_argument("--seed", type=int, default=42)
    simulation.add_argument("--threshold", type=float)
    simulation.add_argument("--json", action="store_true")
    hybrid = commands.add_parser("hybrid", help="Evaluate the Phase 8 local-level state-space model.")
    hybrid.add_argument("path", type=Path, help="Path to the CSV dataset.")
    hybrid.add_argument("--target-column", default="target")
    hybrid.add_argument("--test-size", type=int, default=8)
    hybrid.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "health":
        report = run_health_check()
        if args.json:
            print(json.dumps(report.as_dict(), indent=2))
        else:
            for key, value in report.as_dict().items():
                print(f"{key}: {value}")
        return 0
    if args.command == "neural":
        try:
            _, rows = load_csv(args.path)
            batch = build_sequences_from_rows(rows, args.target_column, args.time_column, args.sequence_length)
            _, losses = train_lstm(batch, input_size=len(batch.sequences[0][0]), epochs=args.epochs)
        except (DataLoadError, KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"model: lstm")
        print(f"sequences: {len(batch.sequences)}")
        print(f"epochs: {args.epochs}")
        print(f"initial_loss: {losses[0]:.6f}")
        print(f"final_loss: {losses[-1]:.6f}")
        return 0
    if args.command == "uncertainty":
        try:
            _, rows = load_csv(args.path)
            if args.coverages:
                levels = [float(value.strip()) for value in args.coverages.split(",") if value.strip()]
                results = evaluate_conformal_levels(rows, levels, args.target_column, args.time_column, args.calibration_size, args.test_size)
            else:
                results = [evaluate_conformal_linear(rows, args.target_column, args.time_column, args.calibration_size, args.test_size, args.coverage)]
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            payload = [result.as_dict() for result in results]
            print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2))
        else:
            for result in results:
                print(f"nominal coverage: {result.nominal_coverage:.0%}")
                print(f"calibration size: {result.calibration_size}")
                print(f"interval radius: {result.radius:.4f}")
                print(f"empirical test coverage: {result.empirical_coverage:.2%}")
                print(f"mean interval width: {result.mean_width:.4f}")
        return 0
    if args.command == "simulate":
        try:
            _, rows = load_csv(args.path)
            result = simulate_dataset(rows, args.target_column, args.horizon, args.paths, args.seed, args.threshold)
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"paths: {result.paths}")
            print(f"horizon: {result.horizon}")
            print(f"median final value: {result.quantiles['0.5'][-1]:.4f}")
            print(f"10%-90% final range: [{result.quantiles['0.1'][-1]:.4f}, {result.quantiles['0.9'][-1]:.4f}]")
            if result.probability_exceeds_threshold is not None:
                print(f"probability of crossing threshold: {result.probability_exceeds_threshold:.2%}")
        return 0
    if args.command == "hybrid":
        try:
            _, rows = load_csv(args.path)
            result = evaluate_local_level(rows, args.target_column, args.test_size)
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"model: {result.model}")
            print(f"train size: {result.train_size}")
            print(f"test size: {result.test_size}")
            print(f"MAE={result.mae:.4f} RMSE={result.rmse:.4f}")
            print(f"interval coverage: {result.interval_coverage:.2%}")
            print(f"mean interval width: {result.mean_interval_width:.4f}")
        return 0
    if args.command == "ml":
        try:
            _, rows = load_csv(args.path)
            scorers = {
                "linear": [evaluate_linear_regression],
                "random-forest": [evaluate_random_forest],
                "all": [evaluate_linear_regression, evaluate_random_forest],
            }[args.model]
            scores = [scorer(rows, args.target_column, args.time_column, args.test_size) for scorer in scorers]
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps([score.as_dict() for score in scores], indent=2))
        else:
            for score in scores:
                print(f"model: {score.model}")
                print(f"train rows: {score.train_rows}")
                print(f"test rows: {score.test_rows}")
                print(f"MAE={score.mae:.4f} RMSE={score.rmse:.4f} MAPE={score.mape:.2f}%")
        return 0
    if args.command == "validate-data":
        try:
            headers, rows = load_csv(args.path)
            validation = validate_rows(headers, rows, time_column=args.time_column, target_column=args.target_column)
            split = chronological_split(rows)
            report = build_report(args.path, headers, rows, validation, train_count=len(split.train), validation_count=len(split.validation), test_count=len(split.test))
        except (DataLoadError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(report.to_json())
        else:
            print(f"status: {'ok' if report.validation_valid else 'invalid'}")
            print(f"rows: {report.row_count}")
            print(f"columns: {', '.join(report.columns)}")
            print(f"split: train={report.train_count}, validation={report.validation_count}, test={report.test_count}")
            print(f"errors: {len(report.errors)}")
            print(f"warnings: {len(report.warnings)}")
            for message in report.errors:
                print(f"error: {message}")
            for message in report.warnings:
                print(f"warning: {message}")
        return 0 if report.validation_valid else 1
    if args.command == "baseline":
        try:
            _, rows = load_csv(args.path)
            values = [float(row[args.target_column]) for row in rows]
            if args.test_size < 1 or args.test_size >= len(values):
                raise ValueError("test-size must be at least 1 and smaller than the dataset")
            scores = evaluate_baselines(values[:-args.test_size], values[-args.test_size:], args.test_size)
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps([score.as_dict() for score in scores], indent=2))
        else:
            print(f"dataset: {args.path}")
            print(f"holdout: {args.test_size} observations")
            for score in scores:
                print(f"{score.model}: MAE={score.mae:.4f} RMSE={score.rmse:.4f} MAPE={score.mape:.2f}%")
        return 0
    print("Unknown command.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

