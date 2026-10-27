"""Command-line entry point for DeepStoch."""

from __future__ import annotations

import argparse
import json
import sys

from deepstoch.core.health import run_health_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepstoch",
        description="DeepStoch uncertainty-aware modelling platform.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health", help="Check the local project environment.")
    health.add_argument("--json", action="store_true", help="Print JSON output.")
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
    print("Unknown command.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

