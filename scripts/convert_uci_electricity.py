"""Convert one UCI ElectricityLoadDiagrams client to DeepStoch CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def convert(source: Path, output: Path, client: str, divide_by_four: bool = False) -> int:
    """Stream a selected client column without loading the 249 MB source at once."""
    if not source.exists():
        raise FileNotFoundError(f"source file does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle, output.open("w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(input_handle, delimiter=";")
        if reader.fieldnames is None or client not in reader.fieldnames:
            raise ValueError(f"client column not found: {client}")
        timestamp_name = reader.fieldnames[0]
        writer = csv.writer(output_handle)
        writer.writerow(["timestamp", "target"])
        for row in reader:
            value = float(row[client])
            writer.writerow([row[timestamp_name].replace(" ", "T"), f"{value / 4 if divide_by_four else value:.6f}"])
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert one UCI electricity client into DeepStoch CSV.")
    parser.add_argument("source", type=Path, help="Path to UCI LD2011_2014.txt")
    parser.add_argument("output", type=Path, help="Canonical DeepStoch CSV output")
    parser.add_argument("--client", required=True, help="Client column name from the source header")
    parser.add_argument("--divide-by-four", action="store_true", help="Convert 15-minute kW readings to kWh as documented by UCI")
    args = parser.parse_args()
    count = convert(args.source, args.output, args.client, args.divide_by_four)
    print(f"converted rows: {count}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
