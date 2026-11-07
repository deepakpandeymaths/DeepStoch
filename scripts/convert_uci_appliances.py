"""Convert UCI Appliances Energy Prediction CSV to DeepStoch format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COVARIATES = ["T1", "RH_1", "T2", "RH_2", "T_out", "RH_out", "Windspeed", "Visibility"]


def convert(source: Path, output: Path, covariates: list[str] | None = None) -> int:
    """Stream the UCI file into timestamp/target plus selected covariates."""
    if not source.exists():
        raise FileNotFoundError(f"source file does not exist: {source}")
    requested = covariates or DEFAULT_COVARIATES
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle, output.open("w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError("source CSV has no header")
        required = {"date", "Appliances"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"source CSV is missing required columns: {', '.join(sorted(missing))}")
        selected = [name for name in requested if name in reader.fieldnames]
        writer = csv.DictWriter(output_handle, fieldnames=["timestamp", "target", *selected])
        writer.writeheader()
        for row in reader:
            output_row = {"timestamp": row["date"].replace(" ", "T"), "target": row["Appliances"]}
            output_row.update({name: row[name] for name in selected})
            writer.writerow(output_row)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert UCI Appliances Energy Prediction CSV.")
    parser.add_argument("source", type=Path, help="Path to energydata_complete.csv")
    parser.add_argument("output", type=Path, help="Canonical DeepStoch CSV output")
    parser.add_argument("--covariates", nargs="*", default=None, help="Optional source columns to preserve")
    args = parser.parse_args()
    count = convert(args.source, args.output, args.covariates)
    print(f"converted rows: {count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
