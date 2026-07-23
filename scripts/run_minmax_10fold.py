"""Run MinMax-scaled 10-fold trial-grouped CV from the existing feature CSV."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.minmax_cv import (  # noqa: E402
    MinMaxCVResult,
    build_minmax_pipelines,
    calculate_fold_metrics,
    calculate_pooled_metrics,
    calculate_trial_per_class_metrics,
    compare_with_subjectwise,
    load_and_validate_feature_table,
    make_trial_grouped_folds,
    run_minmax_trial_grouped_cv,
    save_minmax_cv_artifacts,
)
from src.models import aligned_predict_proba, model_parameter_manifest  # noqa: E402
from src.utils import (  # noqa: E402
    CLASS_NAMES,
    MODEL_SLUGS,
    PROBABILITY_COLUMNS,
    configure_logging,
    file_sha256,
    json_safe,
    runtime_versions,
    set_random_seed,
    stable_json_hash,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/window_features.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/minmax_10fold_trial_grouped"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("saved_models/minmax_10fold_trial_grouped"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify/finalize an already completed 80-fold artifact set without refitting",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def reproduction_command(
    features: Path, output_dir: Path, models_dir: Path, seed: int
) -> str:
    return (
        ".\\.venv\\Scripts\\python.exe scripts\\run_minmax_10fold.py "
        f"--features {_quote(_display_path(features))} "
        f"--output-dir {_quote(_display_path(output_dir))} "
        f"--models-dir {_quote(_display_path(models_dir))} --seed {seed}"
    )


def _snapshot_previous_artifacts(output_dir: Path, models_dir: Path) -> dict[str, str]:
    """Hash only pre-existing subject-wise files, excluding the new subdirectories."""

    candidates = [
        path
        for folder in (PROJECT_ROOT / "outputs", PROJECT_ROOT / "saved_models")
        if folder.exists()
        for path in folder.iterdir()
        if path.is_file()
        and output_dir not in path.parents
        and models_dir not in path.parents
    ]
    return {
        _display_path(path): file_sha256(path)
        for path in sorted(candidates, key=lambda item: str(item).lower())
    }


def load_existing_result(
    output_dir: Path, feature_summary: dict[str, Any]
) -> MinMaxCVResult:
    """Load the tables written before post-run verification (no fitting occurs)."""

    filenames = {
        "oof_window_predictions": "oof_window_predictions.csv",
        "oof_trial_predictions": "oof_trial_predictions.csv",
        "fold_metrics": "fold_metrics.csv",
        "pooled_window_metrics": "pooled_window_metrics.csv",
        "pooled_trial_metrics": "pooled_trial_metrics.csv",
        "per_class_trial_metrics": "per_class_trial_metrics.csv",
        "model_comparison": "model_comparison.csv",
        "fold_participant_overlap": "fold_participant_overlap.csv",
        "fold_assignments": "fold_assignments.csv",
        "fold_model_paths": "fold_model_paths.csv",
    }
    missing = [name for name in filenames.values() if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot verify an incomplete existing result set; missing={missing}"
        )
    loaded = {
        key: pd.read_csv(output_dir / filename)
        for key, filename in filenames.items()
    }
    return MinMaxCVResult(
        **loaded,
        feature_summary=feature_summary,
        artifact_paths={},
    )


def _frames_close(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    key_columns: list[str],
    value_columns: list[str],
    atol: float = 1e-12,
) -> bool:
    left = observed.sort_values(key_columns).reset_index(drop=True)
    right = expected.sort_values(key_columns).reset_index(drop=True)
    return (
        left[key_columns].equals(right[key_columns])
        and np.allclose(
            left[value_columns].to_numpy(float),
            right[value_columns].to_numpy(float),
            atol=atol,
            rtol=1e-10,
            equal_nan=True,
        )
    )


def verify_experiment(
    table: pd.DataFrame,
    feature_names: list[str],
    result: Any,
    output_dir: Path,
    models_dir: Path,
    prior_snapshot: dict[str, str],
) -> dict[str, Any]:
    """Independently validate folds, OOF metrics, scaling scope, and artifacts."""

    checks: dict[str, bool] = {}
    plan = make_trial_grouped_folds(table, seed=42)
    X = table.loc[:, feature_names]
    checks["input_has_7200_rows_and_496_features"] = (
        len(table) == 7200 and len(feature_names) == 496
    )
    checks["only_explicit_eeg_features_enter_X"] = all(
        name.startswith("f__") for name in feature_names
    )
    checks["fold_assignment_covers_every_row_once"] = (
        len(result.fold_assignments) == len(table)
        and result.fold_assignments["row_position"].is_unique
        and set(result.fold_assignments["row_position"]) == set(range(len(table)))
    )
    checks["each_trial_has_one_test_fold"] = bool(
        result.fold_assignments.groupby("trial_id")["fold"].nunique().eq(1).all()
    )
    checks["zero_trial_overlap_all_folds"] = bool(
        result.fold_participant_overlap["trial_overlap_count"].eq(0).all()
    )
    checks["all_four_classes_in_every_test_fold"] = bool(
        result.fold_participant_overlap["all_four_classes_in_test"].all()
    )
    checks["participant_overlap_is_reported_and_allowed"] = bool(
        result.fold_participant_overlap["n_overlapping_participants"].gt(0).all()
        and result.fold_participant_overlap["participant_overlap_allowed"].all()
    )

    windows = result.oof_window_predictions
    trials = result.oof_trial_predictions
    expected_window_rows = len(table) * len(MODEL_SLUGS)
    expected_trial_rows = table["trial_id"].nunique() * len(MODEL_SLUGS)
    checks["complete_unique_oof_window_coverage"] = (
        len(windows) == expected_window_rows
        and not windows.duplicated(["model", "source_row_index"]).any()
        and windows.groupby("model")["source_row_index"].nunique().eq(len(table)).all()
    )
    checks["complete_unique_oof_trial_coverage"] = (
        len(trials) == expected_trial_rows
        and not trials.duplicated(["model", "trial_id"]).any()
        and trials.groupby("model")["trial_id"].nunique().eq(
            table["trial_id"].nunique()
        ).all()
    )
    window_probabilities = windows[PROBABILITY_COLUMNS].to_numpy(float)
    trial_probabilities = trials[PROBABILITY_COLUMNS].to_numpy(float)
    checks["all_probabilities_are_finite_and_normalized"] = bool(
        np.isfinite(window_probabilities).all()
        and np.isfinite(trial_probabilities).all()
        and np.allclose(window_probabilities.sum(axis=1), 1.0, atol=1e-9)
        and np.allclose(trial_probabilities.sum(axis=1), 1.0, atol=1e-9)
    )
    means = (
        windows.groupby(["model", "trial_id"], sort=False)[PROBABILITY_COLUMNS]
        .mean()
        .sort_index()
    )
    saved_means = (
        trials.set_index(["model", "trial_id"])[PROBABILITY_COLUMNS].sort_index()
    )
    checks["trial_probabilities_equal_window_probability_means"] = bool(
        means.index.equals(saved_means.index)
        and np.allclose(means.to_numpy(), saved_means.to_numpy(), atol=1e-12)
    )
    checks["trial_labels_match_mean_probability_argmax"] = np.array_equal(
        trial_probabilities.argmax(axis=1), trials["predicted_label"].to_numpy(int)
    )
    checks["all_trials_have_nine_windows"] = bool(trials["n_windows"].eq(9).all())

    pooled_window = calculate_pooled_metrics(windows)
    pooled_trial = calculate_pooled_metrics(trials)
    metric_columns = [
        "n_samples",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
    ]
    checks["pooled_window_metrics_recompute"] = _frames_close(
        result.pooled_window_metrics,
        pooled_window,
        ["model"],
        metric_columns,
    )
    checks["pooled_trial_metrics_recompute"] = _frames_close(
        result.pooled_trial_metrics,
        pooled_trial,
        ["model"],
        metric_columns,
    )
    recomputed_fold = calculate_fold_metrics(windows, trials)
    fold_value_columns = [
        f"{level}_{metric}"
        for level in ("window", "trial")
        for metric in metric_columns
    ]
    checks["all_80_fold_metric_rows_recompute"] = (
        len(result.fold_metrics) == 80
        and _frames_close(
            result.fold_metrics,
            recomputed_fold,
            ["model", "fold"],
            fold_value_columns,
        )
    )
    recomputed_per_class = calculate_trial_per_class_metrics(trials)
    checks["all_32_per_class_rows_recompute"] = (
        len(result.per_class_trial_metrics) == 32
        and _frames_close(
            result.per_class_trial_metrics,
            recomputed_per_class,
            ["model", "label"],
            ["precision", "recall", "f1", "support"],
        )
    )
    checks["comparison_ranked_by_pooled_trial_macro_f1"] = bool(
        result.model_comparison["pooled_trial_macro_f1"].is_monotonic_decreasing
    )

    scaler_scope_checks: list[bool] = []
    reload_prediction_checks: list[bool] = []
    model_files: list[Path] = []
    for fold, (train_indices, test_indices) in enumerate(plan.splits, start=1):
        train_values = X.iloc[train_indices].to_numpy(dtype=float)
        expected_median = np.nanmedian(train_values, axis=0)
        expected_imputed = np.where(np.isnan(train_values), expected_median, train_values)
        expected_minimum = expected_imputed.min(axis=0)
        expected_maximum = expected_imputed.max(axis=0)
        verification_test = np.asarray(test_indices[:3], dtype=int)
        for slug in MODEL_SLUGS:
            path = models_dir / slug / f"fold_{fold:02d}.joblib"
            model_files.append(path)
            model = joblib.load(path)
            scaler = model.named_steps.get("scaler")
            imputer = model.named_steps.get("imputer")
            scaler_scope_checks.append(
                list(model.named_steps) == ["imputer", "scaler", "classifier"]
                and isinstance(scaler, MinMaxScaler)
                and scaler.feature_range == (0.0, 1.0)
                and int(scaler.n_samples_seen_) == len(train_indices)
                and np.allclose(imputer.statistics_, expected_median, equal_nan=True)
                and np.allclose(scaler.data_min_, expected_minimum)
                and np.allclose(scaler.data_max_, expected_maximum)
                and list(map(str, model.feature_names_in_)) == feature_names
                and np.array_equal(np.asarray(model.classes_, int), np.arange(4))
            )
            expected_rows = (
                windows[
                    windows["model"].eq(slug)
                    & windows["source_row_index"].isin(verification_test)
                ]
                .set_index("source_row_index")
                .loc[verification_test]
            )
            native = np.asarray(model.predict(X.iloc[verification_test]), dtype=int)
            probabilities = aligned_predict_proba(model, X.iloc[verification_test])
            reload_prediction_checks.append(
                np.array_equal(native, expected_rows["predicted_label"].to_numpy(int))
                and np.allclose(
                    probabilities,
                    expected_rows[PROBABILITY_COLUMNS].to_numpy(float),
                    atol=1e-12,
                )
            )
    actual_model_files = set(models_dir.rglob("*.joblib"))
    checks["exactly_80_fold_pipelines_exist"] = (
        len(model_files) == 80
        and set(model_files) == actual_model_files
        and all(path.is_file() for path in model_files)
    )
    checks["all_scalers_fit_only_fold_training_extrema"] = all(scaler_scope_checks)
    checks["reloaded_native_predictions_and_probabilities_match_oof"] = all(
        reload_prediction_checks
    )

    expected_output_files = [
        "model_comparison.csv",
        "model_comparison.png",
        "fold_metrics.csv",
        "pooled_window_metrics.csv",
        "pooled_trial_metrics.csv",
        "per_class_trial_metrics.csv",
        "oof_window_predictions.csv",
        "oof_trial_predictions.csv",
        "fold_participant_overlap.csv",
        "fold_assignments.csv",
        "fold_model_paths.csv",
        "experiment_summary.txt",
        "experiment_manifest.json",
        *[f"confusion_matrix_{slug}.png" for slug in MODEL_SLUGS],
    ]
    previous_metrics_path = PROJECT_ROOT / "outputs" / "model_comparison.csv"
    if previous_metrics_path.is_file():
        expected_output_files.append("comparison_with_subjectwise.csv")
    checks["all_required_output_files_exist"] = all(
        (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in expected_output_files
    )
    if previous_metrics_path.is_file():
        saved_comparison = pd.read_csv(
            output_dir / "comparison_with_subjectwise.csv"
        )
        recomputed_comparison = compare_with_subjectwise(
            previous_metrics_path, result.pooled_trial_metrics
        )
        checks["subjectwise_comparison_recomputes_with_exact_schema"] = (
            list(saved_comparison.columns) == list(recomputed_comparison.columns)
            and len(saved_comparison) == len(MODEL_SLUGS)
            and _frames_close(
                saved_comparison,
                recomputed_comparison,
                ["model"],
                [column for column in saved_comparison.columns if column != "model"],
            )
        )
    saved_model_paths = pd.read_csv(output_dir / "fold_model_paths.csv")
    checks["fold_model_path_table_matches_all_80_files"] = (
        len(saved_model_paths) == 80
        and not saved_model_paths.duplicated(["model", "fold"]).any()
        and set(map(Path, saved_model_paths["model_path"])) == set(model_files)
    )
    current_previous = _snapshot_previous_artifacts(output_dir, models_dir)
    checks["previous_subjectwise_artifacts_unchanged"] = (
        current_previous == prior_snapshot
    )
    previous_manifest_path = PROJECT_ROOT / "outputs" / "reproducibility_manifest.json"
    if previous_manifest_path.is_file():
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
        prior_hashes = previous_manifest.get("artifact_sha256", {})
        checks["previous_subjectwise_manifest_hashes_still_match"] = bool(
            prior_hashes
            and all(
                (PROJECT_ROOT / relative_path).is_file()
                and file_sha256(PROJECT_ROOT / relative_path) == digest
                for relative_path, digest in prior_hashes.items()
            )
        )

    failed = [name for name, passed in checks.items() if not bool(passed)]
    payload = {
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "participant_overlap_counts": result.fold_participant_overlap[
            "n_overlapping_participants"
        ].astype(int).tolist(),
        "average_participant_overlap": float(
            result.fold_participant_overlap["n_overlapping_participants"].mean()
        ),
        "model_file_count": len(model_files),
    }
    write_json(output_dir / "artifact_verification.json", payload)
    if failed:
        raise AssertionError(f"MinMax experiment verification failed: {failed}")
    return payload


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("The predefined experiment requires seed 42")
    features_path = _absolute(args.features)
    output_dir = _absolute(args.output_dir)
    models_dir = _absolute(args.models_dir)
    if output_dir in {PROJECT_ROOT, (PROJECT_ROOT / "outputs").resolve()}:
        raise ValueError(
            "--output-dir must be a dedicated experiment subdirectory, not the "
            "project or previous outputs root"
        )
    if models_dir in {PROJECT_ROOT, (PROJECT_ROOT / "saved_models").resolve()}:
        raise ValueError(
            "--models-dir must be a dedicated experiment subdirectory, not the "
            "project or previous saved_models root"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(None if args.verify_existing else output_dir, args.verbose)
    set_random_seed(args.seed)
    started = datetime.now(timezone.utc)
    command = reproduction_command(features_path, output_dir, models_dir, args.seed)
    previous_metrics = PROJECT_ROOT / "outputs" / "model_comparison.csv"
    prior_snapshot = _snapshot_previous_artifacts(output_dir, models_dir)

    table, feature_names, feature_summary = load_and_validate_feature_table(
        features_path
    )
    fold_plan = make_trial_grouped_folds(table, seed=args.seed)
    class_distribution = {
        CLASS_NAMES[int(label) - 1]: int(count)
        for label, count in table["role_id"].value_counts().sort_index().items()
    }
    print(
        "FEATURE FILE VALIDATION\n"
        f"Rows: {len(table)}; EEG features: {len(feature_names)}; "
        f"X shape: ({len(table)}, {len(feature_names)})\n"
        f"Sessions: {table['session_id'].nunique()}; "
        f"participants: {table['participant_id'].nunique()}; "
        f"trials: {table['trial_id'].nunique()}\n"
        f"Windows/trial: {feature_summary['windows_per_trial']['unique_values']}\n"
        f"Class distribution: {class_distribution}\n"
        f"Missing EEG values: {feature_summary['missing_value_count']}; "
        f"infinite EEG values: {feature_summary['infinite_value_count']}\n"
        f"Metadata columns excluded: {feature_summary['metadata_columns']}\n"
        "Splitter: StratifiedGroupKFold(10, shuffle=True, random_state=42); "
        "group=trial_id\n"
        f"Trial overlap/fold: {fold_plan.participant_overlap['trial_overlap_count'].tolist()}\n"
        f"Participant overlap/fold: {fold_plan.participant_overlap['n_overlapping_participants'].tolist()}"
    )

    if args.verify_existing:
        result = load_existing_result(output_dir, feature_summary)
    else:
        result = run_minmax_trial_grouped_cv(
            table,
            feature_names,
            models_dir=models_dir,
            seed=args.seed,
        )
        save_minmax_cv_artifacts(
            result,
            output_dir,
            previous_subjectwise_metrics=(
                previous_metrics if previous_metrics.is_file() else None
            ),
            reproduction_command=command,
        )
    verification = verify_experiment(
        table,
        feature_names,
        result,
        output_dir,
        models_dir,
        prior_snapshot,
    )

    completed = datetime.now(timezone.utc)
    artifact_files = [
        path
        for path in [*output_dir.iterdir(), *models_dir.rglob("*.joblib")]
        if path.is_file()
        and path.name not in {"run.log", "reproducibility_manifest.json"}
    ]
    templates = build_minmax_pipelines(args.seed)
    write_json(
        output_dir / "reproducibility_manifest.json",
        {
            "status": "complete",
            "finalization_mode": (
                "verification_only_recovery_after_completed_80_fit_run"
                if args.verify_existing
                else "complete_fit_and_verification_run"
            ),
            "started_at_utc": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
            "duration_scope": (
                "post-run verification and manifest finalization only"
                if args.verify_existing
                else "complete feature loading, fitting, evaluation, and verification"
            ),
            "sum_fold_training_seconds": float(
                result.fold_metrics["training_time_seconds"].sum()
            ),
            "sum_fold_prediction_seconds": float(
                result.fold_metrics["prediction_time_seconds"].sum()
            ),
            "training_run_log": _display_path(output_dir / "run.log"),
            "exact_reproduction_command": command,
            "seed": args.seed,
            "feature_source": _display_path(features_path),
            "output_directory": _display_path(output_dir),
            "models_directory": _display_path(models_dir),
            "feature_source_sha256": file_sha256(features_path),
            "feature_summary": feature_summary,
            "splitter": {
                "class": "StratifiedGroupKFold",
                "n_splits": 10,
                "shuffle": True,
                "random_state": 42,
                "group": "trial_id",
            },
            "label_mapping": [
                {"internal_label": index, "source_role_id": index + 1, "class_name": name}
                for index, name in enumerate(CLASS_NAMES)
            ],
            "model_parameters": json_safe(model_parameter_manifest(templates)),
            "artifact_verification": verification,
            "previous_subjectwise_artifact_sha256": prior_snapshot,
            "software_versions": runtime_versions(),
            "configuration_sha256": stable_json_hash(
                {
                    "seed": args.seed,
                    "output_directory": _display_path(output_dir),
                    "models_directory": _display_path(models_dir),
                    "features": feature_names,
                    "models": model_parameter_manifest(templates),
                    "splitter": "StratifiedGroupKFold(10,shuffle=True,random_state=42,groups=trial_id)",
                    "preprocessing": "SimpleImputer(median)+MinMaxScaler(0,1) inside every fold/model",
                }
            ),
            "artifact_sha256": {
                _display_path(path): file_sha256(path)
                for path in sorted(artifact_files, key=lambda item: str(item).lower())
            },
        },
    )
    json.loads(
        (output_dir / "reproducibility_manifest.json").read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON token: {token}")
        ),
    )
    print((output_dir / "experiment_summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
