"""Window- and trial-level evaluation for the fixed classical benchmark.

Window accuracy and macro-F1 are secondary diagnostics.  The primary result is
computed after taking the arithmetic mean of each trial's aligned window class
probabilities and emitting exactly one prediction per original test trial.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .models import INTERNAL_CLASS_LABELS, aligned_predict_proba
from .utils import (
    CLASS_NAMES,
    FEATURE_PREFIX,
    MODEL_SLUGS,
    PROBABILITY_COLUMNS,
    ROLE_NAMES,
    write_json,
)


LOGGER = logging.getLogger(__name__)
EXTERNAL_CLASS_LABELS = np.arange(1, len(CLASS_NAMES) + 1, dtype=int)
IMPORTANCE_MODELS = ("decision_tree", "random_forest", "xgboost")
REQUIRED_WINDOW_METADATA = {
    "window_id",
    "trial_id",
    "session_id",
    "participant_id",
    "question_id",
    "role_id",
    "window_index",
}


@dataclass
class EvaluationArtifacts:
    """In-memory tables and paths created by a complete test evaluation."""

    window_predictions: pd.DataFrame
    trial_predictions: pd.DataFrame
    window_metrics: pd.DataFrame
    trial_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    paths: dict[str, Path]


def _validate_models(models: Mapping[str, Pipeline]) -> None:
    missing = set(MODEL_SLUGS) - set(models)
    extra = set(models) - set(MODEL_SLUGS)
    if missing or extra:
        raise ValueError(
            f"Evaluation requires exactly the eight model slugs; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for slug in MODEL_SLUGS:
        check_is_fitted(models[slug])
        classes = np.asarray(models[slug].classes_, dtype=int)
        if not np.array_equal(classes, INTERNAL_CLASS_LABELS):
            raise ValueError(f"{slug} class order must be [0, 1, 2, 3]")


def _split_window_frame(
    window_features: pd.DataFrame,
    feature_names: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not isinstance(window_features, pd.DataFrame) or window_features.empty:
        raise ValueError("window_features must be a non-empty DataFrame")
    names = (
        [str(name) for name in feature_names]
        if feature_names is not None
        else [str(column) for column in window_features if str(column).startswith(FEATURE_PREFIX)]
    )
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature names must be a non-empty unique ordered list")
    missing_features = set(names) - set(window_features.columns)
    if missing_features:
        raise ValueError(f"Window table is missing features: {sorted(missing_features)[:5]}")
    forbidden = [name for name in names if not name.startswith(FEATURE_PREFIX)]
    if forbidden:
        raise AssertionError(f"Non-signal columns were requested as features: {forbidden[:5]}")

    missing_metadata = REQUIRED_WINDOW_METADATA - set(window_features.columns)
    if missing_metadata:
        raise ValueError(f"Window table is missing metadata: {sorted(missing_metadata)}")
    if window_features["window_id"].duplicated().any():
        raise ValueError("window_id must be unique")
    if window_features.duplicated(["trial_id", "window_index"]).any():
        raise ValueError("Each trial/window_index pair must be unique")

    table = window_features.reset_index(drop=True).copy()
    if "split" in table:
        split_values = set(table["split"].dropna().astype(str).str.casefold())
        if split_values != {"test"}:
            raise AssertionError(
                "Evaluation accepts only untouched test windows; "
                f"received split values {sorted(split_values)}"
            )
    features = table.loc[:, names].apply(pd.to_numeric, errors="raise")
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Test features contain infinite values")
    # NaNs are permitted because the fitted training-only imputer handles them.
    metadata_names = [column for column in table.columns if column not in names]
    metadata = table.loc[:, metadata_names].copy()
    role_ids = metadata["role_id"].to_numpy(dtype=int)
    if not np.isin(role_ids, EXTERNAL_CLASS_LABELS).all():
        raise ValueError("role_id must contain external class labels 1..4")
    if "role_name" in metadata:
        expected_names = np.asarray([ROLE_NAMES[int(label)] for label in role_ids])
        if not np.array_equal(metadata["role_name"].astype(str).to_numpy(), expected_names):
            raise AssertionError("role_name does not match role_id")

    consistency_columns = [
        column
        for column in ("session_id", "participant_id", "question_id", "role_id", "role_name", "split")
        if column in metadata
    ]
    for column in consistency_columns:
        counts = metadata.groupby("trial_id", sort=False)[column].nunique(dropna=False)
        if (counts != 1).any():
            bad = counts[counts != 1].index.tolist()
            raise AssertionError(f"{column} changes within trials: {bad[:5]}")
    return features, metadata, names


def _external_predictions(internal: np.ndarray) -> np.ndarray:
    labels = np.asarray(internal, dtype=int)
    if not np.isin(labels, INTERNAL_CLASS_LABELS).all():
        raise ValueError("Predictions contain unknown internal classes")
    return labels + 1


def _build_window_predictions(
    models: Mapping[str, Pipeline],
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    true_external = metadata["role_id"].to_numpy(dtype=int)
    true_internal = true_external - 1
    tables: list[pd.DataFrame] = []
    prediction_times: dict[str, float] = {}
    for slug in MODEL_SLUGS:
        model = models[slug]
        started = perf_counter()
        probabilities = aligned_predict_proba(model, features)
        # Window-level metrics follow each estimator's native decision rule.
        # In particular, calibrated SVC probabilities are not guaranteed to have
        # the same argmax as SVC.predict(). Trial labels below still follow the
        # required mean-probability rule.
        predicted_internal = np.asarray(model.predict(features), dtype=int)
        if not np.isin(predicted_internal, INTERNAL_CLASS_LABELS).all():
            raise ValueError(f"{slug} returned unknown window prediction labels")
        prediction_times[slug] = float(perf_counter() - started)
        predicted_external = _external_predictions(predicted_internal)
        prediction = metadata.copy()
        prediction.insert(0, "model", slug)
        prediction["true_label"] = true_external
        prediction["true_role"] = [ROLE_NAMES[int(label)] for label in true_external]
        prediction["predicted_label"] = predicted_external
        prediction["predicted_role"] = [
            ROLE_NAMES[int(label)] for label in predicted_external
        ]
        for column_index, column in enumerate(PROBABILITY_COLUMNS):
            prediction[column] = probabilities[:, column_index]
        if not np.array_equal(true_internal, metadata["role_id"].to_numpy(dtype=int) - 1):
            raise AssertionError("Window labels changed during prediction")
        tables.append(prediction)
    combined = pd.concat(tables, ignore_index=True, sort=False)
    expected_rows = len(metadata) * len(MODEL_SLUGS)
    if len(combined) != expected_rows:
        raise AssertionError("Window prediction coverage is incomplete")
    if combined.duplicated(["model", "window_id"]).any():
        raise AssertionError("A model produced duplicate window predictions")
    return combined, prediction_times


def aggregate_window_probabilities(window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Average aligned window probabilities into one prediction per model/trial."""

    required = {
        "model",
        "trial_id",
        "session_id",
        "participant_id",
        "question_id",
        "true_label",
        *PROBABILITY_COLUMNS,
    }
    missing = required - set(window_predictions.columns)
    if missing:
        raise ValueError(f"Window predictions are missing columns: {sorted(missing)}")
    if window_predictions.empty:
        raise ValueError("Cannot aggregate an empty prediction table")
    if "window_id" in window_predictions and window_predictions.duplicated(
        ["model", "window_id"]
    ).any():
        raise ValueError("Duplicate model/window predictions cannot be aggregated")

    identity_candidates = (
        "session_id",
        "participant_id",
        "question_id",
        "role_id",
        "role_name",
        "split",
        "true_label",
        "true_role",
    )
    identity_columns = [
        column for column in identity_candidates if column in window_predictions.columns
    ]
    rows: list[dict[str, Any]] = []
    for (model, trial_id), group in window_predictions.groupby(
        ["model", "trial_id"], sort=False, dropna=False
    ):
        row: dict[str, Any] = {"model": str(model), "trial_id": trial_id}
        for column in identity_columns:
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise AssertionError(f"{column} is inconsistent within trial {trial_id!r}")
            row[column] = values.iloc[0]
        probabilities = group[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < -1e-12).any():
            raise ValueError(f"Trial {trial_id!r} contains invalid window probabilities")
        means = np.mean(probabilities, axis=0)
        total = float(means.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Trial {trial_id!r} has zero probability mass")
        means = np.clip(means / total, 0.0, 1.0)
        means /= means.sum()
        # NumPy argmax provides one deterministic rule for both aggregation and
        # artifact verification: an exact tie selects the first class.
        predicted_external = int(np.argmax(means)) + 1
        row.update(
            {
                "predicted_label": predicted_external,
                "predicted_role": ROLE_NAMES[predicted_external],
                "n_windows": int(len(group)),
                "aggregation": "mean_window_probability",
            }
        )
        row.update(
            {
                column: float(means[index])
                for index, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        rows.append(row)
    trials = pd.DataFrame(rows)
    if trials.duplicated(["model", "trial_id"]).any():
        raise AssertionError("Aggregation produced duplicate trial predictions")
    counts = window_predictions.groupby("model", sort=False)["trial_id"].nunique()
    trial_counts = trials.groupby("model", sort=False)["trial_id"].size()
    if not counts.sort_index().equals(trial_counts.sort_index()):
        raise AssertionError("Aggregation did not preserve model/trial coverage")
    probability_sums = trials[PROBABILITY_COLUMNS].sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(probability_sums, 1.0, atol=1e-8):
        raise AssertionError("Aggregated probabilities do not sum to one")
    return trials


def _metric_record(
    predictions: pd.DataFrame,
    *,
    model: str,
    level: str,
) -> dict[str, Any]:
    true = predictions["true_label"].to_numpy(dtype=int)
    predicted = predictions["predicted_label"].to_numpy(dtype=int)
    probabilities = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    record: dict[str, Any] = {
        "model": model,
        f"n_test_{'windows' if level == 'window' else 'trials'}": int(len(predictions)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(
            f1_score(
                true,
                predicted,
                labels=EXTERNAL_CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }
    if level == "trial":
        record.update(
            {
                "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
                "macro_precision": float(
                    precision_score(
                        true,
                        predicted,
                        labels=EXTERNAL_CLASS_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "macro_recall": float(
                    recall_score(
                        true,
                        predicted,
                        labels=EXTERNAL_CLASS_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        true,
                        predicted,
                        labels=EXTERNAL_CLASS_LABELS,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "cohen_kappa": float(
                    cohen_kappa_score(true, predicted, labels=EXTERNAL_CLASS_LABELS)
                ),
                "matthews_corrcoef": float(matthews_corrcoef(true, predicted)),
                "log_loss": float(
                    log_loss(true, probabilities, labels=EXTERNAL_CLASS_LABELS)
                ),
            }
        )
    return record


def calculate_level_metrics(
    predictions: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    """Calculate one metric row per model at the requested prediction level."""

    if level not in {"window", "trial"}:
        raise ValueError("level must be 'window' or 'trial'")
    rows = []
    for slug in MODEL_SLUGS:
        subset = predictions[predictions["model"] == slug]
        if subset.empty:
            raise ValueError(f"Predictions contain no rows for {slug}")
        rows.append(_metric_record(subset, model=slug, level=level))
    return pd.DataFrame(rows)


def _save_confusion_matrix(predictions: pd.DataFrame, model: str, path: Path) -> None:
    counts = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=EXTERNAL_CLASS_LABELS,
    )
    figure, axis = plt.subplots(figsize=(6.4, 5.5))
    image = axis.imshow(counts, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted role",
        ylabel="True role",
        title=f"{model.replace('_', ' ').title()} — trial-level confusion matrix",
    )
    threshold = counts.max() / 2.0 if counts.size else 0.0
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            axis.text(
                column,
                row,
                str(int(counts[row, column])),
                ha="center",
                va="center",
                color="white" if counts[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_classification_report(predictions: pd.DataFrame, model: str, path: Path) -> None:
    report = classification_report(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=EXTERNAL_CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    path.write_text(
        f"Model: {model}\nEvaluation unit: original trial\n"
        "Aggregation: arithmetic mean of aligned window probabilities\n\n"
        f"{report}",
        encoding="utf-8",
    )


def _save_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    positions = np.arange(len(comparison))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.bar(
        positions - width / 2,
        comparison["trial_accuracy"],
        width,
        label="Trial accuracy",
    )
    axis.bar(
        positions + width / 2,
        comparison["trial_macro_f1"],
        width,
        label="Trial macro F1",
    )
    axis.set_xticks(positions, comparison["model"], rotation=30, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Fixed test-set comparison (primary trial-level metrics)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_feature_importance(
    model: Pipeline,
    model_slug: str,
    feature_names: Sequence[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        raise TypeError(f"{model_slug} does not expose feature_importances_")
    importance = np.asarray(classifier.feature_importances_, dtype=float)
    if importance.shape != (len(feature_names),):
        raise AssertionError(
            f"{model_slug} has {len(importance)} importances for {len(feature_names)} features"
        )
    if not np.isfinite(importance).all() or (importance < 0.0).any():
        raise ValueError(f"{model_slug} returned invalid feature importance")
    total = float(importance.sum())
    normalized = importance / total if total > 0.0 else np.zeros_like(importance)
    importance_type = {
        "decision_tree": "mean_decrease_impurity",
        "random_forest": "mean_decrease_impurity",
        "xgboost": "gain",
    }[model_slug]
    table = pd.DataFrame(
        {
            "model": model_slug,
            "feature_name": list(map(str, feature_names)),
            "importance": importance,
            "importance_normalized": normalized,
            "importance_type": importance_type,
        }
    ).sort_values(["importance", "feature_name"], ascending=[False, True])
    top = table.head(min(20, len(table))).reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1, dtype=int))
    csv_path = output_dir / f"top_features_{model_slug}.csv"
    plot_path = output_dir / f"top_features_{model_slug}.png"
    top.to_csv(csv_path, index=False)

    displayed = top.iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, max(5.0, 0.32 * len(displayed))))
    axis.barh(displayed["feature_name"], displayed["importance_normalized"])
    axis.set_xlabel("Within-model normalized importance")
    axis.set_title(f"{model_slug.replace('_', ' ').title()} — top 20 features")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return csv_path, plot_path


def _comparison_table(
    window_metrics: pd.DataFrame,
    trial_metrics: pd.DataFrame,
    training_times: Mapping[str, float],
    prediction_times: Mapping[str, float],
) -> pd.DataFrame:
    window = window_metrics.rename(
        columns={
            "n_test_windows": "window_n_test_windows",
            "accuracy": "window_accuracy",
            "macro_f1": "window_macro_f1",
        }
    )
    trial = trial_metrics.rename(
        columns={
            column: f"trial_{column}"
            for column in trial_metrics.columns
            if column != "model"
        }
    )
    comparison = trial.merge(window, on="model", how="inner", validate="one_to_one")
    comparison["training_time_seconds"] = comparison["model"].map(training_times)
    comparison["prediction_time_seconds"] = comparison["model"].map(prediction_times)
    if comparison[["training_time_seconds", "prediction_time_seconds"]].isna().any().any():
        raise AssertionError("Missing per-model timing information")
    comparison["rank_trial_macro_f1"] = comparison["trial_macro_f1"].rank(
        method="min", ascending=False
    ).astype(int)
    return comparison.sort_values(
        ["rank_trial_macro_f1", "model"], kind="stable"
    ).reset_index(drop=True)


def evaluate_model_pipelines(
    fitted_models: Mapping[str, Pipeline],
    test_window_features: pd.DataFrame,
    outputs_dir: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
    training_times_seconds: Mapping[str, float] | None = None,
) -> EvaluationArtifacts:
    """Evaluate all fixed models and write every required test artifact."""

    _validate_models(fitted_models)
    features, metadata, ordered_features = _split_window_frame(
        test_window_features, feature_names
    )
    output = Path(outputs_dir)
    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Evaluating eight models on %d test windows from %d trials",
        len(metadata),
        metadata["trial_id"].nunique(),
    )

    window_predictions, prediction_times = _build_window_predictions(
        fitted_models, features, metadata
    )
    trial_predictions = aggregate_window_probabilities(window_predictions)
    expected_trials = metadata["trial_id"].nunique()
    if len(trial_predictions) != expected_trials * len(MODEL_SLUGS):
        raise AssertionError("Trial predictions do not cover every model/test trial")

    window_metrics = calculate_level_metrics(window_predictions, level="window")
    trial_metrics = calculate_level_metrics(trial_predictions, level="trial")
    if training_times_seconds is None:
        raise ValueError("Per-model training timings are required for comparison output")
    missing_times = set(MODEL_SLUGS) - set(training_times_seconds)
    if missing_times:
        raise ValueError(f"Missing training timings for: {sorted(missing_times)}")
    comparison = _comparison_table(
        window_metrics,
        trial_metrics,
        training_times_seconds,
        prediction_times,
    )

    paths: dict[str, Path] = {
        "window_predictions": output / "window_predictions.csv",
        "trial_predictions": output / "trial_predictions.csv",
        "window_metrics": output / "window_level_metrics.csv",
        "trial_metrics": output / "trial_level_metrics.csv",
        "per_class_trial_metrics": output / "per_class_trial_metrics.csv",
        "model_comparison_csv": output / "model_comparison.csv",
        "model_comparison_png": output / "model_comparison.png",
        "metrics_json": output / "metrics.json",
    }
    window_predictions.to_csv(paths["window_predictions"], index=False)
    trial_predictions.to_csv(paths["trial_predictions"], index=False)
    window_metrics.to_csv(paths["window_metrics"], index=False)
    trial_metrics.to_csv(paths["trial_metrics"], index=False)
    per_class_rows: list[dict[str, Any]] = []
    for slug in MODEL_SLUGS:
        subset = trial_predictions[trial_predictions["model"] == slug]
        precision, recall, f1, support = precision_recall_fscore_support(
            subset["true_label"],
            subset["predicted_label"],
            labels=EXTERNAL_CLASS_LABELS,
            zero_division=0,
        )
        for index, label in enumerate(EXTERNAL_CLASS_LABELS):
            per_class_rows.append(
                {
                    "model": slug,
                    "label": int(label),
                    "class_name": CLASS_NAMES[index],
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
    per_class_metrics = pd.DataFrame(per_class_rows)
    per_class_metrics.to_csv(paths["per_class_trial_metrics"], index=False)
    comparison.to_csv(paths["model_comparison_csv"], index=False)
    _save_model_comparison(comparison, paths["model_comparison_png"])

    for slug in MODEL_SLUGS:
        subset = trial_predictions[trial_predictions["model"] == slug]
        confusion_path = output / f"confusion_matrix_{slug}.png"
        report_path = output / f"classification_report_{slug}.txt"
        _save_confusion_matrix(subset, slug, confusion_path)
        _save_classification_report(subset, slug, report_path)
        paths[f"confusion_{slug}"] = confusion_path
        paths[f"classification_report_{slug}"] = report_path

    for slug in IMPORTANCE_MODELS:
        csv_path, plot_path = _save_feature_importance(
            fitted_models[slug], slug, ordered_features, output
        )
        paths[f"importance_csv_{slug}"] = csv_path
        paths[f"importance_png_{slug}"] = plot_path

    write_json(
        paths["metrics_json"],
        {
            "class_order": CLASS_NAMES,
            "internal_class_order": INTERNAL_CLASS_LABELS,
            "external_class_order": EXTERNAL_CLASS_LABELS,
            "primary_evaluation_unit": "trial",
            "trial_aggregation": "unweighted arithmetic mean of aligned window probabilities",
            "secondary_evaluation_unit": "window",
            "window_prediction_rule": "each fitted pipeline's native predict method",
            "n_test_windows": int(len(metadata)),
            "n_test_trials": int(expected_trials),
            "window_level_metrics": window_metrics.to_dict(orient="records"),
            "trial_level_metrics": trial_metrics.to_dict(orient="records"),
            "per_class_trial_metrics": per_class_metrics.to_dict(orient="records"),
            "training_time_seconds": dict(training_times_seconds),
            "prediction_time_seconds": prediction_times,
        },
    )
    return EvaluationArtifacts(
        window_predictions=window_predictions,
        trial_predictions=trial_predictions,
        window_metrics=window_metrics,
        trial_metrics=trial_metrics,
        model_comparison=comparison,
        paths=paths,
    )


# Script-facing aliases keep integration explicit without duplicating behavior.
evaluate_and_save_models = evaluate_model_pipelines
evaluate_models = evaluate_model_pipelines
