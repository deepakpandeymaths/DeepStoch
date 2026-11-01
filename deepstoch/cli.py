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

