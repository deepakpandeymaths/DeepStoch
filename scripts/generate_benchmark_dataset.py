"""Generate a deterministic synthetic dataset for pipeline stress testing."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


def generate(path: Path, rows: int = 720, seed: int = 42) -> None:
    """Create an hourly seasonal dataset; this is not real-world evidence."""
    if rows < 24:
        raise ValueError("rows must be at least 24")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "target", "temperature", "holiday"])
        for index in range(rows):
            timestamp = start + timedelta(hours=index)
            hour = timestamp.hour
            day = index / 24
            temperature = 18 + 7 * math.sin(2 * math.pi * (hour - 5) / 24) + 2 * math.sin(2 * math.pi * day / 30)
            holiday = int(timestamp.weekday() >= 5)
            demand = 430 + 115 * math.sin(2 * math.pi * (hour - 7) / 24) + 35 * math.sin(2 * math.pi * day / 7) - 3.5 * temperature - 25 * holiday + 0.04 * index
            deterministic_noise = 8 * math.sin((index + seed) * 1.618)
            writer.writerow([timestamp.isoformat(), f"{demand + deterministic_noise:.3f}", f"{temperature:.3f}", holiday])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic DeepStoch benchmark dataset.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--rows", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.path, args.rows, args.seed)
    print(f"generated: {args.path}")
    print(f"rows: {args.rows}")
    print("type: synthetic benchmark; not real-world evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
