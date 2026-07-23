"""Execute the complete fixed-split classical EEG machine-learning experiment."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_features import run_feature_stage  # noqa: E402
from scripts.inspect_dataset import inspect_and_split  # noqa: E402
from src.data_loader import (  # noqa: E402
    assert_leakage_free_split,
    assert_no_window_leakage,
    discover_mat_files,
)
from src.evaluation import evaluate_model_pipelines  # noqa: E402
from src.models import (  # noqa: E402
    INTERNAL_CLASS_LABELS,
    aligned_predict_proba,
    fit_model_pipelines_timed,
    model_parameter_manifest,
    save_model_pipelines,
)
from src.utils import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_PREFIX,
    MODEL_SLUGS,
    PROBABILITY_COLUMNS,
    configure_logging,
    file_sha256,
    inventory_hash,
    json_safe,
    runtime_versions,
    session_sort_key,
    set_random_seed,
    stable_json_hash,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def reproduction_command(dataset_root: Path, seed: int) -> str:
    try:
        dataset = dataset_root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        dataset = str(dataset_root)
    if " " in dataset:
        dataset = f'"{dataset}"'
    return (
        ".\\.venv\\Scripts\\python.exe scripts\\run_ml_experiment.py "
        f"--dataset-path {dataset} --seed {seed}"
    )


def _verify_artifacts(
    feature_table: pd.DataFrame,
    feature_names: list[str],
    metadata: pd.DataFrame,
    evaluation: Any,
    model_paths: dict[str, Path],
    outputs_dir: Path,
) -> dict[str, Any]:
    """Recompute saved relationships without refitting or predicting again."""

    checks: dict[str, bool] = {}
    checks["exactly_496_signal_features"] = len(feature_names) == 496
    checks["features_use_explicit_prefix"] = all(
        name.startswith(FEATURE_PREFIX) for name in feature_names
    )
    forbidden_tokens = (
        "session_id",
        "participant_id",
        "trial_id",
        "question_id",
        "window_index",
        "role_id",
        "filename",
        "filepath",
    )
    checks["no_metadata_in_model_features"] = not any(
        any(token in name for token in forbidden_tokens) for name in feature_names
    )
    checks["feature_order_exact"] = list(feature_table[feature_names].columns) == feature_names
    checks["features_have_no_infinity"] = not np.isinf(
        feature_table[feature_names].to_numpy(dtype=float)
    ).any()

    train = feature_table[feature_table["split"] == "train"]
    test = feature_table[feature_table["split"] == "test"]
    checks["train_test_trials_disjoint"] = set(train["trial_id"]).isdisjoint(
        test["trial_id"]
    )
    checks["train_test_participants_disjoint"] = set(
        train["participant_id"]
    ).isdisjoint(test["participant_id"])
    checks["all_retained_trials_have_nine_windows"] = bool(
        feature_table.groupby("trial_id")["window_index"].nunique().eq(9).all()
    )

    window_predictions = evaluation.window_predictions
    trial_predictions = evaluation.trial_predictions
    probabilities_window = window_predictions[PROBABILITY_COLUMNS].to_numpy(float)
    probabilities_trial = trial_predictions[PROBABILITY_COLUMNS].to_numpy(float)
    expected_test_windows = len(test)
    expected_test_trials = test["trial_id"].nunique()
    checks["all_models_have_window_predictions"] = (
        len(window_predictions) == expected_test_windows * len(MODEL_SLUGS)
        and set(window_predictions["model"]) == set(MODEL_SLUGS)
    )
    checks["all_models_have_trial_predictions"] = (
        len(trial_predictions) == expected_test_trials * len(MODEL_SLUGS)
        and set(trial_predictions["model"]) == set(MODEL_SLUGS)
    )
    checks["window_prediction_keys_unique"] = not window_predictions.duplicated(
        ["model", "window_id"]
    ).any()
    checks["trial_prediction_keys_unique"] = not trial_predictions.duplicated(
        ["model", "trial_id"]
    ).any()
    checks["probabilities_finite"] = bool(
        np.isfinite(probabilities_window).all() and np.isfinite(probabilities_trial).all()
    )
    checks["probabilities_sum_to_one"] = bool(
        np.allclose(probabilities_window.sum(axis=1), 1.0, atol=1e-8)
        and np.allclose(probabilities_trial.sum(axis=1), 1.0, atol=1e-8)
    )
    checks["trial_predictions_match_probability_argmax"] = bool(
        np.array_equal(
            probabilities_trial.argmax(axis=1) + 1,
            trial_predictions["predicted_label"].to_numpy(int),
        )
    )
    checks["trial_window_count_is_nine"] = bool(trial_predictions["n_windows"].eq(9).all())

    recomputed = (
        window_predictions.groupby(["model", "trial_id"], sort=False)[
            PROBABILITY_COLUMNS
        ]
        .mean()
        .sort_index()
    )
    saved = (
        trial_predictions.set_index(["model", "trial_id"])[PROBABILITY_COLUMNS]
        .sort_index()
    )
    checks["trial_probabilities_recompute_from_windows"] = bool(
        recomputed.index.equals(saved.index)
        and np.allclose(recomputed.to_numpy(), saved.to_numpy(), atol=1e-10)
    )

    metric_checks = []
    for _, row in evaluation.trial_metrics.iterrows():
        subset = trial_predictions[trial_predictions["model"] == row["model"]]
        metric_checks.append(
            np.isclose(
                accuracy_score(subset["true_label"], subset["predicted_label"]),
                row["accuracy"],
            )
            and np.isclose(
                f1_score(
                    subset["true_label"],
                    subset["predicted_label"],
                    labels=[1, 2, 3, 4],
                    average="macro",
                    zero_division=0,
                ),
                row["macro_f1"],
            )
        )
    checks["trial_metrics_recompute"] = all(metric_checks)
    checks["comparison_sorted_by_trial_macro_f1"] = bool(
        evaluation.model_comparison["trial_macro_f1"].is_monotonic_decreasing
    )

    reload_checks = []
    reload_probability_checks = []
    reload_prediction_checks = []
    verification_rows = test.head(min(8, len(test)))
    for slug in MODEL_SLUGS:
        path = model_paths[slug]
        model = joblib.load(path)
        reload_checks.append(
            path.name == f"{slug}.joblib"
            and np.array_equal(np.asarray(model.classes_, int), INTERNAL_CLASS_LABELS)
            and list(map(str, model.feature_names_in_)) == feature_names
        )
        reloaded_probabilities = aligned_predict_proba(
            model, verification_rows.loc[:, feature_names]
        )
        expected_probabilities = (
            window_predictions[
                window_predictions["model"].eq(slug)
                & window_predictions["window_id"].isin(verification_rows["window_id"])
            ]
            .set_index("window_id")
            .loc[verification_rows["window_id"], PROBABILITY_COLUMNS]
            .to_numpy(float)
        )
        reload_probability_checks.append(
            np.allclose(reloaded_probabilities, expected_probabilities, atol=1e-12)
        )
        reloaded_predictions = np.asarray(
            model.predict(verification_rows.loc[:, feature_names]), dtype=int
        ) + 1
        expected_predictions = (
            window_predictions[
                window_predictions["model"].eq(slug)
                & window_predictions["window_id"].isin(verification_rows["window_id"])
            ]
            .set_index("window_id")
            .loc[verification_rows["window_id"], "predicted_label"]
            .to_numpy(int)
        )
        reload_prediction_checks.append(
            np.array_equal(reloaded_predictions, expected_predictions)
        )
    checks["all_eight_joblib_pipelines_reload"] = all(reload_checks)
    checks["reloaded_probabilities_match_saved_predictions"] = all(
        reload_probability_checks
    )
    checks["reloaded_native_predictions_match_saved_window_labels"] = all(
        reload_prediction_checks
    )
    checks["all_confusion_matrices_exist"] = all(
        (outputs_dir / f"confusion_matrix_{slug}.png").stat().st_size > 0
        for slug in MODEL_SLUGS
    )
    checks["tree_importance_artifacts_have_top20"] = all(
        len(pd.read_csv(outputs_dir / f"top_features_{slug}.csv")) == 20
        and (outputs_dir / f"top_features_{slug}.png").stat().st_size > 0
        for slug in ("decision_tree", "random_forest", "xgboost")
    )
    checks["per_class_metrics_have_32_rows"] = (
        len(pd.read_csv(outputs_dir / "per_class_trial_metrics.csv")) == 32
    )
    checks["raw_split_still_leakage_free"] = bool(
        assert_leakage_free_split(metadata)["passed"]
    )

    failed = [name for name, passed in checks.items() if not bool(passed)]
    payload = {"passed": not failed, "checks": checks, "failed": failed}
    write_json(outputs_dir / "artifact_verification.json", payload)
    if failed:
        raise AssertionError(f"Artifact verification failed: {failed}")
    return payload


def _clean_json(value: Any) -> Any:
    return json_safe(value)


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("The requested reproducible experiment seed is fixed at 42")
    set_random_seed(args.seed)
    outputs = PROJECT_ROOT / "outputs"
    cache = PROJECT_ROOT / "cache"
    saved_models = PROJECT_ROOT / "saved_models"
    for directory in (outputs, cache, saved_models):
        directory.mkdir(parents=True, exist_ok=True)
    configure_logging(outputs, args.verbose)
    started_at = datetime.now(timezone.utc)

    metadata, split_summary, dataset_root, paper = inspect_and_split(
        args.dataset_path, outputs, args.seed
    )
    metadata, feature_table, feature_names, feature_manifest = run_feature_stage(
        metadata, PROJECT_ROOT, outputs, cache
    )
    assert_no_window_leakage(feature_table)

    train_rows = feature_table[feature_table["split"] == "train"].copy()
    test_rows = feature_table[feature_table["split"] == "test"].copy()
    X_train = train_rows.loc[:, feature_names]
    y_train = train_rows["role_id"].to_numpy(dtype=int) - 1
    if set(np.unique(y_train)) != {0, 1, 2, 3}:
        raise AssertionError("Retained training windows do not contain all four classes")
    if set(test_rows["role_id"].astype(int)) != {1, 2, 3, 4}:
        raise AssertionError("Retained test windows do not contain all four classes")

    fitted_models, training_times = fit_model_pipelines_timed(
        X_train,
        y_train,
        feature_names=feature_names,
        seed=args.seed,
    )
    model_paths = save_model_pipelines(fitted_models, saved_models)
    evaluation = evaluate_model_pipelines(
        fitted_models,
        test_rows,
        outputs,
        feature_names=feature_names,
        training_times_seconds=training_times,
    )
    verification = _verify_artifacts(
        feature_table,
        feature_names,
        metadata,
        evaluation,
        model_paths,
        outputs,
    )

    comparison = evaluation.model_comparison
    best = comparison.iloc[0]
    raw_train_trials = metadata[metadata["split"].eq("train")]
    raw_test_trials = metadata[metadata["split"].eq("test")]
    train_trials = metadata[
        metadata["preprocessing_included"].astype(bool)
        & metadata["split"].eq("train")
    ]
    test_trials = metadata[
        metadata["preprocessing_included"].astype(bool)
        & metadata["split"].eq("test")
    ]
    train_sessions = sorted(raw_train_trials["session_id"].unique(), key=session_sort_key)
    test_sessions = sorted(raw_test_trials["session_id"].unique(), key=session_sort_key)
    rejected_train = len(raw_train_trials) - len(train_trials)
    rejected_test = len(raw_test_trials) - len(test_trials)
    command = reproduction_command(dataset_root, args.seed)
    best_model_path = model_paths[str(best["model"])]
    lines = [
        "FINAL CLASSICAL-ML EXPERIMENT SUMMARY",
        f"Training sessions: {len(train_sessions)} ({100.0 * len(train_sessions) / 17:.2f}%; {', '.join(train_sessions)})",
        f"Testing sessions: {len(test_sessions)} ({100.0 * len(test_sessions) / 17:.2f}%; {', '.join(test_sessions)})",
        f"Training participants assigned: {raw_train_trials['participant_id'].nunique()} ({100.0 * raw_train_trials['participant_id'].nunique() / metadata['participant_id'].nunique():.2f}%)",
        f"Testing participants assigned: {raw_test_trials['participant_id'].nunique()} ({100.0 * raw_test_trials['participant_id'].nunique() / metadata['participant_id'].nunique():.2f}%)",
        f"Training trials assigned: {len(raw_train_trials)} ({100.0 * len(raw_train_trials) / len(metadata):.2f}%)",
        f"Testing trials assigned: {len(raw_test_trials)} ({100.0 * len(raw_test_trials) / len(metadata):.2f}%)",
        f"Training trials analyzed: {len(train_trials)} ({100.0 * len(train_trials) / (len(train_trials) + len(test_trials)):.2f}% of retained trials)",
        f"Testing trials analyzed: {len(test_trials)} ({100.0 * len(test_trials) / (len(train_trials) + len(test_trials)):.2f}% of retained trials)",
        f"Training windows: {len(train_rows)} ({100.0 * len(train_rows) / len(feature_table):.2f}%)",
        f"Testing windows: {len(test_rows)} ({100.0 * len(test_rows) / len(feature_table):.2f}%)",
        f"Rejected trials: {rejected_train + rejected_test} (train={rejected_train}, test={rejected_test})",
        f"Best model (descriptive test ranking): {best['model']}",
        f"Trial accuracy: {best['trial_accuracy']:.4f}",
        f"Trial macro F1: {best['trial_macro_f1']:.4f}",
        f"Trial weighted F1: {best['trial_weighted_f1']:.4f}",
        f"Cohen's kappa: {best['trial_cohen_kappa']:.4f}",
        "Participant overlap: 0",
        f"Results table: {outputs / 'model_comparison.csv'}",
        f"Best model pipeline: {best_model_path}",
        f"Reproduce: {command}",
        "Note: window metrics are secondary because overlapping windows within a trial are correlated.",
        "Note: model ranking is descriptive and was not used for tuning or retraining.",
    ]
    final_summary = "\n".join(lines) + "\n"
    (outputs / "final_summary.txt").write_text(final_summary, encoding="utf-8")

    completed_at = datetime.now(timezone.utc)
    model_params = _clean_json(model_parameter_manifest(fitted_models))
    artifact_files = [
        path
        for folder in (outputs, saved_models)
        for path in folder.glob("*")
        if path.is_file() and path.name not in {"run.log", "reproducibility_manifest.json"}
    ]
    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "exact_reproduction_command": command,
        "seed": args.seed,
        "paper_path": str(paper),
        "dataset_path": str(dataset_root),
        "dataset_inventory_sha256": inventory_hash(
            discover_mat_files(dataset_root), dataset_root
        ),
        "group_split": split_summary,
        "class_order": [
            {"external_label": index + 1, "internal_label": index, "class_name": name}
            for index, name in enumerate(CLASS_NAMES)
        ],
        "feature_stage": feature_manifest,
        "ordered_feature_names": feature_names,
        "model_parameters": model_params,
        "training_time_seconds": training_times,
        "counts": {
            "discovered_trials": len(metadata),
            "rejected_trials": int((~metadata["preprocessing_included"].astype(bool)).sum()),
            "raw_assigned_training_sessions": len(train_sessions),
            "raw_assigned_testing_sessions": len(test_sessions),
            "raw_assigned_training_participants": raw_train_trials["participant_id"].nunique(),
            "raw_assigned_testing_participants": raw_test_trials["participant_id"].nunique(),
            "raw_assigned_training_trials": len(raw_train_trials),
            "raw_assigned_testing_trials": len(raw_test_trials),
            "rejected_training_trials": rejected_train,
            "rejected_testing_trials": rejected_test,
            "retained_training_participants": train_trials["participant_id"].nunique(),
            "retained_testing_participants": test_trials["participant_id"].nunique(),
            "analyzed_training_trials": len(train_trials),
            "analyzed_testing_trials": len(test_trials),
            "training_windows": len(train_rows),
            "testing_windows": len(test_rows),
        },
        "actual_percentages": {
            "training_sessions": 100.0 * len(train_sessions) / 17,
            "testing_sessions": 100.0 * len(test_sessions) / 17,
            "training_participants": 100.0
            * raw_train_trials["participant_id"].nunique()
            / metadata["participant_id"].nunique(),
            "testing_participants": 100.0
            * raw_test_trials["participant_id"].nunique()
            / metadata["participant_id"].nunique(),
            "raw_assigned_training_trials": 100.0 * len(raw_train_trials) / len(metadata),
            "raw_assigned_testing_trials": 100.0 * len(raw_test_trials) / len(metadata),
            "training_retained_trials": 100.0
            * len(train_trials)
            / (len(train_trials) + len(test_trials)),
            "testing_retained_trials": 100.0
            * len(test_trials)
            / (len(train_trials) + len(test_trials)),
            "training_windows": 100.0 * len(train_rows) / len(feature_table),
            "testing_windows": 100.0 * len(test_rows) / len(feature_table),
        },
        "fit_scope": "all imputers, scalers, and classifiers fitted on training windows only",
        "primary_evaluation": "trial-level mean window probability",
        "secondary_evaluation": "correlated window-level predictions",
        "best_model_is_descriptive_not_retrained": str(best["model"]),
        "artifact_verification": verification,
        "software_versions": runtime_versions(),
        "configuration_sha256": stable_json_hash(
            {
                "seed": args.seed,
                "split": split_summary,
                "features": feature_names,
                "models": model_params,
            }
        ),
        "artifact_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in sorted(artifact_files)
        },
    }
    write_json(outputs / "reproducibility_manifest.json", run_manifest)
    print(final_summary)


if __name__ == "__main__":
    main()
