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
    print("Unknown command.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

