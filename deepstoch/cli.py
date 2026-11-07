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
from deepstoch.experiments.hybrid import evaluate_local_level, evaluate_local_trend, evaluate_ml_state_hybrid
from deepstoch.experiments.rolling import evaluate_repeated_rolling_neural_state, evaluate_rolling_ml_state, evaluate_rolling_neural_state
from deepstoch.models.neural_state import train_neural_state_space
from deepstoch.experiments.neural_eval import evaluate_neural_state_space
from deepstoch.experiments.compare import compare_models


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
    hybrid.add_argument("--model", choices=["level", "trend", "ml-state", "all"], default="all")
    hybrid.add_argument("--time-column", default="timestamp")
    hybrid.add_argument("--json", action="store_true")
    rolling = commands.add_parser("rolling-hybrid", help="Run rolling-origin evaluation for the ML-state hybrid.")
    rolling.add_argument("path", type=Path, help="Path to the CSV dataset.")
    rolling.add_argument("--time-column", default="timestamp")
    rolling.add_argument("--target-column", default="target")
    rolling.add_argument("--minimum-train", type=int, default=24)
    rolling.add_argument("--test-size", type=int, default=4)
    rolling.add_argument("--step", type=int, default=4)
    rolling.add_argument("--json", action="store_true")
    rolling_neural = commands.add_parser("rolling-neural", help="Run leakage-safe rolling-origin neural state-space evaluation.")
    rolling_neural.add_argument("path", type=Path, help="Path to the CSV dataset.")
    rolling_neural.add_argument("--time-column", default="timestamp")
    rolling_neural.add_argument("--target-column", default="target")
    rolling_neural.add_argument("--sequence-length", type=int, default=6)
    rolling_neural.add_argument("--minimum-train", type=int, default=120)
    rolling_neural.add_argument("--test-size", type=int, default=24)
    rolling_neural.add_argument("--step", type=int, default=24)
    rolling_neural.add_argument("--epochs", type=int, default=20)
    rolling_neural.add_argument("--patience", type=int, default=5)
    rolling_neural.add_argument("--coverage", type=float, default=0.9)
    rolling_neural.add_argument("--json", action="store_true")
    repeated_neural = commands.add_parser("rolling-neural-repeated", help="Repeat rolling neural evaluation across random seeds.")
    repeated_neural.add_argument("path", type=Path, help="Path to the CSV dataset.")
    repeated_neural.add_argument("--time-column", default="timestamp")
    repeated_neural.add_argument("--target-column", default="target")
    repeated_neural.add_argument("--sequence-length", type=int, default=6)
    repeated_neural.add_argument("--minimum-train", type=int, default=120)
    repeated_neural.add_argument("--test-size", type=int, default=24)
    repeated_neural.add_argument("--step", type=int, default=24)
    repeated_neural.add_argument("--epochs", type=int, default=20)
    repeated_neural.add_argument("--patience", type=int, default=5)
    repeated_neural.add_argument("--coverage", type=float, default=0.9)
    repeated_neural.add_argument("--seeds", default="42,43,44", help="Comma-separated unique random seeds.")
    repeated_neural.add_argument("--json", action="store_true")
    neural_state = commands.add_parser("neural-state", help="Train the optional Phase 8 neural state-space model.")
    neural_state.add_argument("path", type=Path, help="Path to the CSV dataset.")
    neural_state.add_argument("--time-column", default="timestamp")
    neural_state.add_argument("--target-column", default="target")
    neural_state.add_argument("--sequence-length", type=int, default=6)
    neural_state.add_argument("--epochs", type=int, default=50)
    neural_state.add_argument("--patience", type=int, default=10)
    neural_state.add_argument("--checkpoint", type=Path, default=Path("artifacts/neural_state_space.pt"))
    neural_eval = commands.add_parser("neural-evaluate", help="Evaluate a trained neural state-space checkpoint.")
    neural_eval.add_argument("path", type=Path, help="Path to the CSV dataset.")
    neural_eval.add_argument("--checkpoint", type=Path, default=Path("artifacts/neural_state_space.pt"))
    neural_eval.add_argument("--time-column", default="timestamp")
    neural_eval.add_argument("--target-column", default="target")
    neural_eval.add_argument("--sequence-length", type=int, default=6)
    neural_eval.add_argument("--test-size", type=int, default=8)
    neural_eval.add_argument("--coverage", type=float, default=0.9)
    neural_eval.add_argument("--json", action="store_true")
    compare = commands.add_parser("compare", help="Build a unified model comparison report.")
    compare.add_argument("path", type=Path, help="Path to the CSV dataset.")
    compare.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint trained on this same dataset.")
    compare.add_argument("--time-column", default="timestamp")
    compare.add_argument("--target-column", default="target")
    compare.add_argument("--test-size", type=int, default=8)
    compare.add_argument("--json", action="store_true")
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
            evaluators = {"level": [evaluate_local_level], "trend": [evaluate_local_trend], "ml-state": [evaluate_ml_state_hybrid], "all": [evaluate_local_level, evaluate_local_trend, evaluate_ml_state_hybrid]}[args.model]
            results = [evaluator(rows, args.target_column, args.time_column, args.test_size) if evaluator is evaluate_ml_state_hybrid else evaluator(rows, args.target_column, args.test_size) for evaluator in evaluators]
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps([result.as_dict() for result in results], indent=2))
        else:
            for result in results:
                print(f"model: {result.model}")
                print(f"train size: {result.train_size}")
                print(f"test size: {result.test_size}")
                print(f"MAE={result.mae:.4f} RMSE={result.rmse:.4f}")
                print(f"interval coverage: {result.interval_coverage:.2%}")
                print(f"mean interval width: {result.mean_interval_width:.4f}")
        return 0
    if args.command == "rolling-hybrid":
        try:
            _, rows = load_csv(args.path)
            result = evaluate_rolling_ml_state(rows, args.target_column, args.time_column, minimum_train=args.minimum_train, test_size=args.test_size, step=args.step)
        except (DataLoadError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"model: {result.model}")
            print(f"folds: {result.folds}")
            print(f"test size: {result.test_size}")
            print(f"mean RMSE: {result.mean_rmse:.4f}")
            print(f"RMSE standard deviation: {result.std_rmse:.4f}")
            print(f"mean interval coverage: {result.mean_coverage:.2%}")
            print(f"mean interval width: {result.mean_interval_width:.4f}")
        return 0
    if args.command == "rolling-neural":
        try:
            _, rows = load_csv(args.path)
            result = evaluate_rolling_neural_state(
                rows,
                args.target_column,
                args.time_column,
                sequence_length=args.sequence_length,
                minimum_train=args.minimum_train,
                test_size=args.test_size,
                step=args.step,
                epochs=args.epochs,
                patience=args.patience,
                nominal_coverage=args.coverage,
            )
        except (DataLoadError, KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"model: {result.model}")
            print(f"folds: {result.folds}")
            print(f"test size: {result.test_size}")
            print(f"mean RMSE: {result.mean_rmse:.4f}")
            print(f"RMSE standard deviation: {result.std_rmse:.4f}")
            print(f"mean interval coverage: {result.mean_coverage:.2%}")
            print(f"mean interval width: {result.mean_interval_width:.4f}")
        return 0
    if args.command == "rolling-neural-repeated":
        try:
            _, rows = load_csv(args.path)
            seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            result = evaluate_repeated_rolling_neural_state(
                rows,
                args.target_column,
                args.time_column,
                seeds=seeds,
                sequence_length=args.sequence_length,
                minimum_train=args.minimum_train,
                test_size=args.test_size,
                step=args.step,
                epochs=args.epochs,
                patience=args.patience,
                nominal_coverage=args.coverage,
            )
        except (DataLoadError, KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"model: {result.model}")
            print(f"seeds: {','.join(str(seed) for seed in result.seeds)}")
            print(f"folds per seed: {result.folds_per_seed}")
            print(f"mean RMSE: {result.mean_rmse:.4f}")
            print(f"RMSE standard deviation: {result.std_rmse:.4f}")
            print(f"mean interval coverage: {result.mean_coverage:.2%}")
            print(f"coverage standard deviation: {result.std_coverage:.2%}")
            print(f"absolute coverage error: {result.calibration['absolute_coverage_error']:.2%}")
            print(f"fold coverage range: [{result.calibration['minimum_fold_coverage']:.2%}, {result.calibration['maximum_fold_coverage']:.2%}]")
            print(f"mean interval width: {result.mean_interval_width:.4f}")
        return 0
    if args.command == "neural-state":
        try:
            _, rows = load_csv(args.path)
            batch = build_sequences_from_rows(rows, args.target_column, args.time_column, args.sequence_length)
            training_batch, validation_batch = batch.split()
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _, losses, validation_losses = train_neural_state_space(training_batch, input_size=len(batch.sequences[0][0]), validation_batch=validation_batch, epochs=args.epochs, patience=args.patience, checkpoint_path=args.checkpoint)
        except (DataLoadError, KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("model: neural_state_space")
        print(f"sequences: {len(batch.sequences)}")
        print(f"training sequences: {len(training_batch.sequences)}")
        print(f"validation sequences: {len(validation_batch.sequences)}")
        print(f"epochs: {args.epochs}")
        print(f"initial_nll: {losses[0]:.6f}")
        print(f"final_nll: {losses[-1]:.6f}")
        if validation_losses:
            print(f"best_validation_nll: {min(validation_losses):.6f}")
        print(f"checkpoint: {args.checkpoint}")
        return 0
    if args.command == "neural-evaluate":
        try:
            _, rows = load_csv(args.path)
            result = evaluate_neural_state_space(rows, args.checkpoint, args.target_column, args.time_column, args.sequence_length, args.test_size, args.coverage)
        except (DataLoadError, KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(f"model: {result.model}")
            print(f"test size: {result.test_size}")
            print(f"MAE={result.mae:.4f} RMSE={result.rmse:.4f} MAPE={result.mape:.2f}%")
            print(f"nominal coverage: {result.nominal_coverage:.0%}")
            print(f"empirical coverage: {result.empirical_coverage:.2%}")
            print(f"mean interval width: {result.mean_interval_width:.4f}")
        return 0
    if args.command == "compare":
        try:
            _, rows = load_csv(args.path)
            result = compare_models(rows, target_column=args.target_column, time_column=args.time_column, test_size=args.test_size, neural_checkpoint=args.checkpoint)
        except (DataLoadError, KeyError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps([row.as_dict() for row in result], indent=2))
        else:
            print("model | RMSE | MAE | MAPE | coverage | interval width")
            for row in result:
                print(f"{row.model} | {row.rmse if row.rmse is not None else '-'} | {row.mae if row.mae is not None else '-'} | {row.mape if row.mape is not None else '-'} | {row.coverage if row.coverage is not None else '-'} | {row.interval_width if row.interval_width is not None else '-'}")
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

