"""Trial-grouped 10-fold MinMax cross-validation for window EEG features.

This module is intentionally separate from the completed subject-wise experiment.
It consumes the existing ``window_features.csv`` table, never refits EEG feature
extraction, and keeps every window from an original trial in one test fold.  The
participant overlap permitted by this non-subject-wise experiment is measured and
reported rather than treated as leakage under this experiment's stated design.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .models import INTERNAL_CLASS_LABELS, aligned_predict_proba, build_model_pipelines
from .utils import (
    CLASS_NAMES,
    FEATURE_PREFIX,
    MODEL_SLUGS,
    PROBABILITY_COLUMNS,
    RANDOM_SEED,
    ROLE_NAMES,
    write_json,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_N_SPLITS = 10
EXTERNAL_CLASS_LABELS = np.arange(1, len(CLASS_NAMES) + 1, dtype=int)
REQUIRED_METADATA_COLUMNS = {
    "window_id",
    "trial_id",
    "session_id",
    "participant_id",
    "question_id",
    "window_index",
    "role_id",
}


@dataclass(frozen=True)
class TrialGroupedFoldPlan:
    """Validated CV indices and their auditable group assignments."""

    splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    assignments: pd.DataFrame
    participant_overlap: pd.DataFrame
    n_splits: int
    seed: int


@dataclass
class MinMaxCVResult:
    """All OOF predictions and summaries from the eight fixed procedures."""

    oof_window_predictions: pd.DataFrame
    oof_trial_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    pooled_window_metrics: pd.DataFrame
    pooled_trial_metrics: pd.DataFrame
    per_class_trial_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    fold_participant_overlap: pd.DataFrame
    fold_assignments: pd.DataFrame
    feature_summary: dict[str, Any]
    fold_model_paths: pd.DataFrame
    artifact_paths: dict[str, Path]


def _ordered_feature_names(
    table: pd.DataFrame,
    feature_names: Sequence[str] | None,
) -> list[str]:
    names = (
        [str(name) for name in feature_names]
        if feature_names is not None
        else [str(column) for column in table.columns if str(column).startswith(FEATURE_PREFIX)]
    )
    if not names:
        raise ValueError(f"No EEG feature columns beginning with {FEATURE_PREFIX!r} were found")
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    missing = set(names) - set(table.columns)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)[:10]}")
    forbidden = [name for name in names if not name.startswith(FEATURE_PREFIX)]
    if forbidden:
        raise AssertionError(f"Metadata columns cannot enter X: {forbidden[:10]}")
    return names


def validate_window_feature_table(
    table: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[list[str], dict[str, Any]]:
    """Validate one row/window and return the exact signal feature order and summary."""

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Window feature table must be a non-empty pandas DataFrame")
    missing_metadata = REQUIRED_METADATA_COLUMNS - set(table.columns)
    if missing_metadata:
        raise ValueError(f"Window feature table is missing {sorted(missing_metadata)}")
    names = _ordered_feature_names(table, feature_names)
    if table["window_id"].isna().any() or table["window_id"].duplicated().any():
        raise AssertionError("window_id values must be present and globally unique")
    if table[["trial_id", "session_id", "participant_id", "question_id"]].isna().any().any():
        raise AssertionError("Grouping metadata contain missing values")
    if table.duplicated(["trial_id", "window_index"]).any():
        raise AssertionError("A trial/window_index pair occurs more than once")

    role_ids = pd.to_numeric(table["role_id"], errors="raise").to_numpy(dtype=int)
    if not np.isin(role_ids, EXTERNAL_CLASS_LABELS).all():
        raise ValueError("role_id must use external labels 1, 2, 3, and 4")
    if set(np.unique(role_ids)) != set(EXTERNAL_CLASS_LABELS):
        raise ValueError("The feature table must contain all four behavioral roles")
    if "role_name" in table:
        expected = np.asarray([ROLE_NAMES[int(label)] for label in role_ids])
        if not np.array_equal(table["role_name"].astype(str).to_numpy(), expected):
            raise AssertionError("role_name is inconsistent with role_id")

    constant_within_trial = [
        column
        for column in (
            "session_id",
            "participant_id",
            "question_id",
            "role_id",
            "role_name",
        )
        if column in table
    ]
    for column in constant_within_trial:
        counts = table.groupby("trial_id", sort=False)[column].nunique(dropna=False)
        if (counts != 1).any():
            bad = counts[counts != 1].index.tolist()
            raise AssertionError(f"{column} changes within trial IDs: {bad[:5]}")

    features = table.loc[:, names]
    nonnumeric = [name for name in names if not pd.api.types.is_numeric_dtype(features[name])]
    if nonnumeric:
        raise TypeError(f"EEG features must be numeric: {nonnumeric[:10]}")
    feature_array = features.to_numpy(dtype=float)
    infinite_count = int(np.isinf(feature_array).sum())
    if infinite_count:
        raise ValueError(
            f"EEG feature matrix contains {infinite_count} infinite values; "
            "the median imputer cannot handle infinity"
        )

    trial_labels = table.groupby("trial_id", sort=False)["role_id"].first().astype(int)
    trial_class_counts = trial_labels.value_counts().reindex(EXTERNAL_CLASS_LABELS, fill_value=0)
    if int(trial_class_counts.min()) < n_splits:
        raise ValueError(
            f"At least {n_splits} trial groups per class are required; "
            f"observed {trial_class_counts.to_dict()}"
        )
    windows_per_trial = table.groupby("trial_id", sort=False).size()
    window_class_counts = (
        pd.Series(role_ids).value_counts().reindex(EXTERNAL_CLASS_LABELS, fill_value=0)
    )
    missing_by_feature = features.isna().sum()
    summary: dict[str, Any] = {
        "row_count": int(len(table)),
        "feature_count": int(len(names)),
        "class_order": {
            str(index): name for index, name in enumerate(CLASS_NAMES)
        },
        "window_class_distribution_external_labels": {
            str(int(label)): int(window_class_counts.loc[label])
            for label in EXTERNAL_CLASS_LABELS
        },
        "trial_class_distribution_external_labels": {
            str(int(label)): int(trial_class_counts.loc[label])
            for label in EXTERNAL_CLASS_LABELS
        },
        "participant_count": int(table["participant_id"].nunique()),
        "session_count": int(table["session_id"].nunique()),
        "trial_count": int(table["trial_id"].nunique()),
        "windows_per_trial": {
            "minimum": int(windows_per_trial.min()),
            "maximum": int(windows_per_trial.max()),
            "unique_values": sorted(map(int, windows_per_trial.unique())),
            "frequency": {
                str(int(count)): int(frequency)
                for count, frequency in windows_per_trial.value_counts().sort_index().items()
            },
        },
        "missing_value_count": int(missing_by_feature.sum()),
        "features_with_missing_values": int((missing_by_feature > 0).sum()),
        "missing_values_by_feature": {
            str(name): int(count)
            for name, count in missing_by_feature[missing_by_feature > 0].items()
        },
        "infinite_value_count": infinite_count,
        "metadata_columns": [column for column in table.columns if column not in names],
        "feature_columns": names,
        "prior_split_column_ignored": "split" in table.columns,
    }
    return names, summary


def load_and_validate_feature_table(
    path: str | Path,
    feature_names: Sequence[str] | None = None,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load the existing feature CSV without changing its row order."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {source}")
    table = pd.read_csv(source)
    names, summary = validate_window_feature_table(
        table, feature_names, n_splits=n_splits
    )
    summary["source_path"] = str(source)
    return table, names, summary


def make_trial_grouped_folds(
    table: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = RANDOM_SEED,
) -> TrialGroupedFoldPlan:
    """Create exactly one deterministic StratifiedGroupKFold test fold per trial."""

    validate_window_feature_table(table, n_splits=n_splits)
    if n_splits != DEFAULT_N_SPLITS:
        raise ValueError("The predefined experiment requires exactly 10 folds")
    if seed != RANDOM_SEED:
        raise ValueError("The predefined experiment requires random_state=42")

    labels = table["role_id"].to_numpy(dtype=int) - 1
    groups = table["trial_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    raw_splits = list(splitter.split(np.zeros(len(table)), labels, groups))
    if len(raw_splits) != n_splits:
        raise AssertionError(f"Expected {n_splits} folds, observed {len(raw_splits)}")

    test_counts = np.zeros(len(table), dtype=int)
    assignment_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    class_groups = (
        table[["trial_id", "role_id"]]
        .drop_duplicates("trial_id")
        .groupby("role_id")["trial_id"]
        .nunique()
    )
    class_complete_is_possible = bool(
        class_groups.reindex(EXTERNAL_CLASS_LABELS, fill_value=0).ge(n_splits).all()
    )

    for fold, (train_indices, test_indices) in enumerate(raw_splits, start=1):
        train = np.asarray(train_indices, dtype=int)
        test = np.asarray(test_indices, dtype=int)
        train_trials = set(groups[train])
        test_trials = set(groups[test])
        trial_overlap = train_trials & test_trials
        if trial_overlap:
            raise AssertionError(
                f"Fold {fold} separates trial groups: {sorted(trial_overlap)[:5]}"
            )
        train_classes = set(labels[train])
        test_classes = set(labels[test])
        if train_classes != set(INTERNAL_CLASS_LABELS):
            raise AssertionError(f"Fold {fold} training data are missing classes")
        if class_complete_is_possible and test_classes != set(INTERNAL_CLASS_LABELS):
            raise AssertionError(
                f"Fold {fold} test data are missing classes despite sufficient groups"
            )

        train_participants = set(table.iloc[train]["participant_id"].astype(str))
        test_participants = set(table.iloc[test]["participant_id"].astype(str))
        overlapping = sorted(train_participants & test_participants)
        train_sessions = set(table.iloc[train]["session_id"].astype(str))
        test_sessions = set(table.iloc[test]["session_id"].astype(str))
        overlap_rows.append(
            {
                "fold": fold,
                "n_train_windows": int(len(train)),
                "n_test_windows": int(len(test)),
                "n_train_trials": int(len(train_trials)),
                "n_test_trials": int(len(test_trials)),
                "n_train_participants": int(len(train_participants)),
                "n_test_participants": int(len(test_participants)),
                "n_overlapping_participants": int(len(overlapping)),
                "participant_overlap_fraction_of_test": float(
                    len(overlapping) / len(test_participants)
                ),
                "overlapping_participant_ids": ";".join(overlapping),
                "participant_overlap_allowed": True,
                "n_overlapping_sessions": int(len(train_sessions & test_sessions)),
                "trial_overlap_count": 0,
                "test_classes_internal": ";".join(map(str, sorted(test_classes))),
                "all_four_classes_in_test": test_classes == set(INTERNAL_CLASS_LABELS),
                **{
                    f"n_test_trials_{CLASS_NAMES[label].lower()}": int(
                        table.iloc[test]
                        .loc[lambda frame: frame["role_id"].eq(label + 1), "trial_id"]
                        .nunique()
                    )
                    for label in INTERNAL_CLASS_LABELS
                },
            }
        )
        for row_index in test:
            row = table.iloc[int(row_index)]
            assignment_rows.append(
                {
                    "row_position": int(row_index),
                    "window_id": row["window_id"],
                    "trial_id": row["trial_id"],
                    "session_id": row["session_id"],
                    "participant_id": row["participant_id"],
                    "role_id": int(row["role_id"]),
                    "true_label": int(row["role_id"]) - 1,
                    "fold": fold,
                }
            )
        test_counts[test] += 1
        splits.append((train, test))

    if not np.all(test_counts == 1):
        bad = np.flatnonzero(test_counts != 1)
        raise AssertionError(
            "Every window row must be tested exactly once; "
            f"bad row positions: {bad[:10].tolist()}"
        )
    assignments = pd.DataFrame(assignment_rows).sort_values("row_position").reset_index(drop=True)
    if len(assignments) != len(table) or assignments["window_id"].duplicated().any():
        raise AssertionError("Fold assignments do not cover every window exactly once")
    if assignments.groupby("trial_id")["fold"].nunique().ne(1).any():
        raise AssertionError("A trial was assigned to more than one test fold")
    return TrialGroupedFoldPlan(
        splits=tuple(splits),
        assignments=assignments,
        participant_overlap=pd.DataFrame(overlap_rows),
        n_splits=n_splits,
        seed=seed,
    )


def build_minmax_pipelines(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Clone the eight fixed classifiers into imputer/MinMax fold pipelines."""

    if seed != RANDOM_SEED:
        raise ValueError("The predefined model configurations require random_state=42")
    fixed = build_model_pipelines(seed)
    pipelines: dict[str, Pipeline] = {}
    for slug in MODEL_SLUGS:
        classifier = clone(fixed[slug].named_steps["classifier"])
        pipelines[slug] = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", MinMaxScaler(feature_range=(0.0, 1.0))),
                ("classifier", classifier),
            ]
        )
    return pipelines


def _identity_metadata(table: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    names = set(feature_names)
    metadata = table.loc[:, [column for column in table.columns if column not in names]].copy()
    if "split" in metadata:
        metadata = metadata.rename(columns={"split": "previous_subjectwise_split"})
    metadata = metadata.rename(columns={"role_id": "source_role_id"})
    metadata.insert(0, "source_row_index", np.arange(len(metadata), dtype=int))
    return metadata


def _window_prediction_rows(
    model_slug: str,
    fold: int,
    model: Pipeline,
    X_test: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = aligned_predict_proba(model, X_test)
    predicted_internal = np.asarray(model.predict(X_test), dtype=int)
    if not np.isin(predicted_internal, INTERNAL_CLASS_LABELS).all():
        raise ValueError(f"{model_slug} fold {fold} returned invalid native predictions")
    true_internal = test_metadata["source_role_id"].to_numpy(dtype=int) - 1
    output = test_metadata.reset_index(drop=True).copy()
    output.insert(0, "fold", int(fold))
    output.insert(0, "model", model_slug)
    output["true_label"] = true_internal
    output["true_role"] = [CLASS_NAMES[int(label)] for label in true_internal]
    output["predicted_label"] = predicted_internal
    output["predicted_role"] = [CLASS_NAMES[int(label)] for label in predicted_internal]
    for column_index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probabilities[:, column_index]
    return output


def aggregate_oof_trial_probabilities(oof_window_predictions: pd.DataFrame) -> pd.DataFrame:
    """Average window probabilities into one OOF row per model/original trial."""

    required = {
        "model",
        "fold",
        "trial_id",
        "session_id",
        "participant_id",
        "question_id",
        "true_label",
        *PROBABILITY_COLUMNS,
    }
    missing = required - set(oof_window_predictions.columns)
    if missing:
        raise ValueError(f"OOF window predictions are missing {sorted(missing)}")
    identity_candidates = (
        "fold",
        "session_id",
        "participant_id",
        "question_id",
        "source_role_id",
        "role_name",
        "previous_subjectwise_split",
        "true_label",
        "true_role",
    )
    identity_columns = [
        column for column in identity_candidates if column in oof_window_predictions
    ]
    rows: list[dict[str, Any]] = []
    for (model_slug, trial_id), group in oof_window_predictions.groupby(
        ["model", "trial_id"], sort=False, dropna=False
    ):
        row: dict[str, Any] = {"model": str(model_slug), "trial_id": trial_id}
        for column in identity_columns:
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise AssertionError(f"{column} changes within OOF trial {trial_id!r}")
            row[column] = values.iloc[0]
        probabilities = group[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < -1e-12).any():
            raise ValueError(f"Invalid probabilities for trial {trial_id!r}")
        mean_probability = probabilities.mean(axis=0)
        total = float(mean_probability.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(f"Zero probability mass for trial {trial_id!r}")
        mean_probability = np.clip(mean_probability / total, 0.0, 1.0)
        mean_probability /= mean_probability.sum()
        predicted_internal = int(np.argmax(mean_probability))
        row.update(
            {
                "predicted_label": predicted_internal,
                "predicted_role": CLASS_NAMES[predicted_internal],
                "n_windows": int(len(group)),
                "aggregation": "mean_window_probability",
            }
        )
        row.update(
            {
                column: float(mean_probability[index])
                for index, column in enumerate(PROBABILITY_COLUMNS)
            }
        )
        rows.append(row)
    trials = pd.DataFrame(rows)
    if trials.duplicated(["model", "trial_id"]).any():
        raise AssertionError("A trial received more than one OOF prediction per model")
    expected = oof_window_predictions.groupby("model")["trial_id"].nunique().sort_index()
    observed = trials.groupby("model")["trial_id"].size().sort_index()
    if not expected.equals(observed):
        raise AssertionError("Trial aggregation lost OOF coverage")
    return trials


def _classification_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    true = predictions["true_label"].to_numpy(dtype=int)
    predicted = predictions["predicted_label"].to_numpy(dtype=int)
    return {
        "n_samples": int(len(predictions)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(
            precision_score(
                true,
                predicted,
                labels=INTERNAL_CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true,
                predicted,
                labels=INTERNAL_CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true,
                predicted,
                labels=INTERNAL_CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true,
                predicted,
                labels=INTERNAL_CLASS_LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(true, predicted, labels=INTERNAL_CLASS_LABELS)
        ),
    }


def calculate_fold_metrics(
    oof_window_predictions: pd.DataFrame,
    oof_trial_predictions: pd.DataFrame,
    timing: Mapping[tuple[str, int], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Return one row/model/fold with window- and trial-level metrics."""

    rows: list[dict[str, Any]] = []
    for slug in MODEL_SLUGS:
        for fold in range(1, DEFAULT_N_SPLITS + 1):
            row: dict[str, Any] = {"model": slug, "fold": fold}
            for level, predictions in (
                ("window", oof_window_predictions),
                ("trial", oof_trial_predictions),
            ):
                subset = predictions[
                    predictions["model"].eq(slug) & predictions["fold"].eq(fold)
                ]
                if subset.empty:
                    raise AssertionError(f"No {level} predictions for {slug} fold {fold}")
                for metric, value in _classification_metrics(subset).items():
                    row[f"{level}_{metric}"] = value
            if timing is not None and (slug, fold) in timing:
                row["training_time_seconds"] = float(timing[(slug, fold)][0])
                row["prediction_time_seconds"] = float(timing[(slug, fold)][1])
            rows.append(row)
    return pd.DataFrame(rows)


def calculate_pooled_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one pooled OOF metric row per canonical model."""

    rows = []
    for slug in MODEL_SLUGS:
        subset = predictions[predictions["model"].eq(slug)]
        if subset.empty:
            raise AssertionError(f"No pooled OOF predictions for {slug}")
        rows.append({"model": slug, **_classification_metrics(subset)})
    return pd.DataFrame(rows)


def calculate_trial_per_class_metrics(oof_trial_predictions: pd.DataFrame) -> pd.DataFrame:
    """Return pooled precision/recall/F1/support for four roles and eight models."""

    rows: list[dict[str, Any]] = []
    for slug in MODEL_SLUGS:
        subset = oof_trial_predictions[oof_trial_predictions["model"].eq(slug)]
        precision, recall, f1, support = precision_recall_fscore_support(
            subset["true_label"],
            subset["predicted_label"],
            labels=INTERNAL_CLASS_LABELS,
            zero_division=0,
        )
        for index, label in enumerate(INTERNAL_CLASS_LABELS):
            rows.append(
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
    return pd.DataFrame(rows)


def build_model_comparison(
    pooled_window_metrics: pd.DataFrame,
    pooled_trial_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Combine pooled and fold distributions, ranked by pooled trial macro-F1."""

    window = pooled_window_metrics.rename(
        columns={
            column: f"pooled_window_{column}"
            for column in pooled_window_metrics.columns
            if column != "model"
        }
    )
    trial = pooled_trial_metrics.rename(
        columns={
            column: f"pooled_trial_{column}"
            for column in pooled_trial_metrics.columns
            if column != "model"
        }
    )
    metric_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
    )
    summaries = []
    for slug in MODEL_SLUGS:
        row: dict[str, Any] = {"model": slug}
        for evaluation_level in ("window", "trial"):
            group = fold_metrics[fold_metrics["model"].eq(slug)]
            if len(group) != DEFAULT_N_SPLITS:
                raise AssertionError(
                    f"Expected {DEFAULT_N_SPLITS} {evaluation_level}-level fold rows "
                    f"for {slug}, found {len(group)}"
                )
            for metric in metric_names:
                row[f"{evaluation_level}_fold_{metric}_mean"] = float(
                    group[f"{evaluation_level}_{metric}"].mean()
                )
                row[f"{evaluation_level}_fold_{metric}_std"] = float(
                    group[f"{evaluation_level}_{metric}"].std(ddof=1)
                )
        summaries.append(row)
    fold_summary = pd.DataFrame(summaries)
    comparison = (
        trial.merge(window, on="model", validate="one_to_one")
        .merge(fold_summary, on="model", validate="one_to_one")
    )
    comparison["rank_pooled_trial_macro_f1"] = comparison[
        "pooled_trial_macro_f1"
    ].rank(method="min", ascending=False).astype(int)
    return comparison.sort_values(
        ["rank_pooled_trial_macro_f1", "model"], kind="stable"
    ).reset_index(drop=True)


def run_minmax_trial_grouped_cv(
    table: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    previous_subjectwise_metrics: str | Path | pd.DataFrame | None = None,
    seed: int = RANDOM_SEED,
) -> MinMaxCVResult:
    """Execute the fixed 8×10 OOF procedure; no hyperparameter selection occurs."""

    names, summary = validate_window_feature_table(table, feature_names)
    plan = make_trial_grouped_folds(table, seed=seed)
    X = table.loc[:, names]
    y = table["role_id"].to_numpy(dtype=int) - 1
    metadata = _identity_metadata(table, names)
    model_root = Path(models_dir) if models_dir is not None else None
    if model_root is not None:
        model_root.mkdir(parents=True, exist_ok=True)

    prediction_tables: list[pd.DataFrame] = []
    path_rows: list[dict[str, Any]] = []
    timing: dict[tuple[str, int], tuple[float, float]] = {}
    pipeline_templates = build_minmax_pipelines(seed)
    for slug in MODEL_SLUGS:
        for fold, (train_indices, test_indices) in enumerate(plan.splits, start=1):
            LOGGER.info("MinMax 10-fold: fitting %s fold %02d", slug, fold)
            pipeline = clone(pipeline_templates[slug])
            train = np.asarray(train_indices, dtype=int)
            test = np.asarray(test_indices, dtype=int)
            if set(np.unique(y[train])) != set(INTERNAL_CLASS_LABELS):
                raise AssertionError(f"{slug} fold {fold} training classes are incomplete")
            started = perf_counter()
            try:
                pipeline.fit(X.iloc[train], y[train])
            except Exception as error:
                raise RuntimeError(f"{slug} failed while fitting fold {fold}") from error
            training_seconds = float(perf_counter() - started)
            scaler = pipeline.named_steps["scaler"]
            if int(scaler.n_samples_seen_) != len(train):
                raise AssertionError(
                    f"{slug} fold {fold} MinMaxScaler was not fitted on train rows only"
                )
            if not np.array_equal(np.asarray(pipeline.classes_, dtype=int), INTERNAL_CLASS_LABELS):
                raise AssertionError(f"{slug} fold {fold} fitted class order is not 0..3")

            started = perf_counter()
            fold_predictions = _window_prediction_rows(
                slug,
                fold,
                pipeline,
                X.iloc[test],
                metadata.iloc[test],
            )
            prediction_seconds = float(perf_counter() - started)
            timing[(slug, fold)] = (training_seconds, prediction_seconds)
            prediction_tables.append(fold_predictions)
            if model_root is not None:
                model_directory = model_root / slug
                model_directory.mkdir(parents=True, exist_ok=True)
                model_path = model_directory / f"fold_{fold:02d}.joblib"
                joblib.dump(pipeline, model_path)
                path_rows.append(
                    {"model": slug, "fold": fold, "model_path": str(model_path)}
                )
            del pipeline

    oof_windows = pd.concat(prediction_tables, ignore_index=True, sort=False)
    expected_window_rows = len(table) * len(MODEL_SLUGS)
    if len(oof_windows) != expected_window_rows:
        raise AssertionError(
            f"Expected {expected_window_rows} OOF window predictions, got {len(oof_windows)}"
        )
    if oof_windows.duplicated(["model", "window_id"]).any():
        raise AssertionError("A model/window received more than one OOF prediction")
    for slug in MODEL_SLUGS:
        if set(oof_windows.loc[oof_windows["model"].eq(slug), "window_id"]) != set(
            table["window_id"]
        ):
            raise AssertionError(f"{slug} does not cover every window exactly once")

    oof_trials = aggregate_oof_trial_probabilities(oof_windows)
    expected_trial_rows = table["trial_id"].nunique() * len(MODEL_SLUGS)
    if len(oof_trials) != expected_trial_rows:
        raise AssertionError(
            f"Expected {expected_trial_rows} OOF trial predictions, got {len(oof_trials)}"
        )
    fold_metrics = calculate_fold_metrics(oof_windows, oof_trials, timing)
    fold_metrics = fold_metrics.merge(
        plan.participant_overlap[
            [
                "fold",
                "n_train_windows",
                "n_test_windows",
                "n_train_trials",
                "n_test_trials",
                "n_overlapping_participants",
            ]
        ],
        on="fold",
        how="left",
        validate="many_to_one",
    )
    pooled_window = calculate_pooled_metrics(oof_windows)
    pooled_trial = calculate_pooled_metrics(oof_trials)
    per_class = calculate_trial_per_class_metrics(oof_trials)
    comparison = build_model_comparison(pooled_window, pooled_trial, fold_metrics)
    result = MinMaxCVResult(
        oof_window_predictions=oof_windows,
        oof_trial_predictions=oof_trials,
        fold_metrics=fold_metrics,
        pooled_window_metrics=pooled_window,
        pooled_trial_metrics=pooled_trial,
        per_class_trial_metrics=per_class,
        model_comparison=comparison,
        fold_participant_overlap=plan.participant_overlap,
        fold_assignments=plan.assignments,
        feature_summary=summary,
        fold_model_paths=pd.DataFrame(path_rows),
        artifact_paths={},
    )
    if output_dir is not None:
        result.artifact_paths = save_minmax_cv_artifacts(
            result,
            output_dir,
            previous_subjectwise_metrics=previous_subjectwise_metrics,
        )
    return result


def _save_confusion_matrix(predictions: pd.DataFrame, slug: str, path: Path) -> None:
    counts = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=INTERNAL_CLASS_LABELS,
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
        title=f"{slug.replace('_', ' ').title()} — pooled OOF trial confusion",
    )
    threshold = counts.max() / 2.0 if counts.size else 0.0
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
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


def _save_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    positions = np.arange(len(comparison))
    width = 0.25
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for offset, column, label in (
        (-width, "pooled_trial_accuracy", "Trial accuracy"),
        (0.0, "pooled_trial_macro_f1", "Trial macro F1"),
        (width, "pooled_trial_cohen_kappa", "Cohen's kappa"),
    ):
        axis.bar(positions + offset, comparison[column], width, label=label)
    axis.set_xticks(positions, comparison["model"], rotation=30, ha="right")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Pooled OOF score")
    axis.set_title("MinMax 10-fold trial-grouped comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_with_subjectwise(
    previous: str | Path | pd.DataFrame,
    pooled_trial_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build the explicitly non-controlled comparison requested by the protocol."""

    old = pd.read_csv(previous) if isinstance(previous, (str, Path)) else previous.copy()
    if "model" not in old:
        raise ValueError("Previous metrics require a model column")
    if old["model"].duplicated().any() or set(old["model"]) != set(MODEL_SLUGS):
        raise ValueError("Previous metrics must contain each canonical model exactly once")
    if (
        pooled_trial_metrics["model"].duplicated().any()
        or set(pooled_trial_metrics["model"]) != set(MODEL_SLUGS)
    ):
        raise ValueError("New pooled metrics must contain each canonical model exactly once")

    def old_column(*candidates: str) -> str:
        for candidate in candidates:
            if candidate in old:
                return candidate
        raise ValueError(f"Previous metrics are missing one of {candidates}")

    selected = old[
        [
            "model",
            old_column("accuracy", "trial_accuracy"),
            old_column("macro_f1", "trial_macro_f1"),
            old_column("cohen_kappa", "trial_cohen_kappa"),
        ]
    ].copy()
    selected.columns = [
        "model",
        "previous_subjectwise_trial_accuracy",
        "previous_subjectwise_macro_f1",
        "previous_subjectwise_kappa",
    ]
    current = pooled_trial_metrics[
        ["model", "accuracy", "macro_f1", "cohen_kappa"]
    ].rename(
        columns={
            "accuracy": "new_non_subjectwise_10fold_trial_accuracy",
            "macro_f1": "new_non_subjectwise_macro_f1",
            "cohen_kappa": "new_non_subjectwise_kappa",
        }
    )
    comparison = selected.merge(current, on="model", how="inner", validate="one_to_one")
    if len(comparison) != len(MODEL_SLUGS):
        raise AssertionError("Subject-wise comparison lost model coverage")
    return comparison


def _experiment_summary_text(
    result: MinMaxCVResult,
    output_dir: Path,
    reproduction_command: str | None,
) -> str:
    best = result.model_comparison.iloc[0]
    trial_fold_rows = result.fold_metrics[
        result.fold_metrics["model"].eq(best["model"])
    ]
    class_distribution = result.feature_summary[
        "window_class_distribution_external_labels"
    ]
    lines = [
        "MINMAX 10-FOLD TRIAL-GROUPED EXPERIMENT SUMMARY",
        f"Feature-matrix shape: ({result.feature_summary['row_count']}, {result.feature_summary['feature_count']})",
        f"Unique trials: {result.feature_summary['trial_count']}",
        f"Unique participants: {result.feature_summary['participant_count']}",
        f"Unique sessions: {result.feature_summary['session_count']}",
        f"Window class distribution (external labels): {class_distribution}",
        "Splitter: StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)",
        "Trial overlap in every fold: 0",
        f"Average overlapping participants/fold: {result.fold_participant_overlap['n_overlapping_participants'].mean():.2f}",
        f"Best model by pooled trial macro F1: {best['model']}",
        f"Pooled trial accuracy: {best['pooled_trial_accuracy']:.6f}",
        f"Pooled trial macro F1: {best['pooled_trial_macro_f1']:.6f}",
        f"Pooled trial weighted F1: {best['pooled_trial_weighted_f1']:.6f}",
        f"Pooled trial Cohen's kappa: {best['pooled_trial_cohen_kappa']:.6f}",
        f"Best-model trial fold accuracy mean ± std: {trial_fold_rows['trial_accuracy'].mean():.6f} ± {trial_fold_rows['trial_accuracy'].std(ddof=1):.6f}",
        f"Best-model trial fold macro F1 mean ± std: {trial_fold_rows['trial_macro_f1'].mean():.6f} ± {trial_fold_rows['trial_macro_f1'].std(ddof=1):.6f}",
        f"Best-model trial fold weighted F1 mean ± std: {trial_fold_rows['trial_weighted_f1'].mean():.6f} ± {trial_fold_rows['trial_weighted_f1'].std(ddof=1):.6f}",
        f"Best-model trial fold kappa mean ± std: {trial_fold_rows['trial_cohen_kappa'].mean():.6f} ± {trial_fold_rows['trial_cohen_kappa'].std(ddof=1):.6f}",
        f"Output directory: {output_dir}",
        "Participant overlap is permitted here; these scores do not demonstrate generalization to unseen participants.",
        "This is not a controlled scaler comparison: StandardScaler was replaced with MinMaxScaler for the five previously scaled models (and unscaled tree models now also use MinMaxScaler).",
        "At the same time, the fixed subject-wise holdout was replaced with participant-overlapping, trial-grouped 10-fold cross-validation.",
    ]
    if reproduction_command:
        lines.append(f"Reproduce: {reproduction_command}")
    return "\n".join(lines) + "\n"


def save_minmax_cv_artifacts(
    result: MinMaxCVResult,
    output_dir: str | Path,
    *,
    previous_subjectwise_metrics: str | Path | pd.DataFrame | None = None,
    reproduction_command: str | None = None,
) -> dict[str, Path]:
    """Write the new experiment's artifacts without touching subject-wise outputs."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "model_comparison_csv": destination / "model_comparison.csv",
        "model_comparison_png": destination / "model_comparison.png",
        "fold_metrics": destination / "fold_metrics.csv",
        "pooled_window_metrics": destination / "pooled_window_metrics.csv",
        "pooled_trial_metrics": destination / "pooled_trial_metrics.csv",
        "per_class_trial_metrics": destination / "per_class_trial_metrics.csv",
        "oof_window_predictions": destination / "oof_window_predictions.csv",
        "oof_trial_predictions": destination / "oof_trial_predictions.csv",
        "fold_participant_overlap": destination / "fold_participant_overlap.csv",
        "experiment_summary": destination / "experiment_summary.txt",
        "experiment_manifest": destination / "experiment_manifest.json",
        "fold_assignments": destination / "fold_assignments.csv",
        "fold_model_paths": destination / "fold_model_paths.csv",
    }
    result.model_comparison.to_csv(paths["model_comparison_csv"], index=False)
    result.fold_metrics.to_csv(paths["fold_metrics"], index=False)
    result.pooled_window_metrics.to_csv(paths["pooled_window_metrics"], index=False)
    result.pooled_trial_metrics.to_csv(paths["pooled_trial_metrics"], index=False)
    result.per_class_trial_metrics.to_csv(paths["per_class_trial_metrics"], index=False)
    result.oof_window_predictions.to_csv(paths["oof_window_predictions"], index=False)
    result.oof_trial_predictions.to_csv(paths["oof_trial_predictions"], index=False)
    result.fold_participant_overlap.to_csv(paths["fold_participant_overlap"], index=False)
    result.fold_assignments.to_csv(paths["fold_assignments"], index=False)
    result.fold_model_paths.to_csv(paths["fold_model_paths"], index=False)
    _save_model_comparison(result.model_comparison, paths["model_comparison_png"])

    for slug in MODEL_SLUGS:
        confusion_path = destination / f"confusion_matrix_{slug}.png"
        subset = result.oof_trial_predictions[
            result.oof_trial_predictions["model"].eq(slug)
        ]
        _save_confusion_matrix(subset, slug, confusion_path)
        paths[f"confusion_matrix_{slug}"] = confusion_path

    if previous_subjectwise_metrics is not None:
        comparison = compare_with_subjectwise(
            previous_subjectwise_metrics, result.pooled_trial_metrics
        )
        comparison_path = destination / "comparison_with_subjectwise.csv"
        comparison.to_csv(comparison_path, index=False)
        paths["comparison_with_subjectwise"] = comparison_path

    summary_text = _experiment_summary_text(
        result, destination, reproduction_command
    )
    paths["experiment_summary"].write_text(summary_text, encoding="utf-8")
    write_json(
        paths["experiment_manifest"],
        {
            "experiment": "minmax_10fold_trial_grouped",
            "splitter": {
                "class": "StratifiedGroupKFold",
                "n_splits": 10,
                "shuffle": True,
                "random_state": 42,
                "group": "trial_id",
                "stratification_label": "role_id converted to internal 0..3",
            },
            "scaling": "SimpleImputer(median) then MinMaxScaler(0,1), fitted inside each fold for every model",
            "feature_summary": result.feature_summary,
            "fold_participant_overlap": result.fold_participant_overlap.to_dict(
                orient="records"
            ),
            "primary_ranking_metric": "pooled OOF trial macro F1",
            "participant_overlap_warning": (
                "Participants may occur in training and testing; results are not "
                "evidence of unseen-participant generalization."
            ),
            "comparison_warning": (
                "The five previously scaled models changed StandardScaler to MinMaxScaler; "
                "the three tree models changed unscaled to MinMaxScaler; and the evaluation "
                "changed from a fixed subject-wise holdout to participant-overlapping grouped "
                "CV. This is not a controlled scaler comparison."
            ),
            "fold_model_paths": result.fold_model_paths.to_dict(orient="records"),
            "reproduction_command": reproduction_command,
        },
    )
    result.artifact_paths = paths
    return paths


# Descriptive aliases for scripts and tests.
build_trial_grouped_folds = make_trial_grouped_folds
aggregate_trial_probabilities = aggregate_oof_trial_probabilities
run_cross_validation = run_minmax_trial_grouped_cv
save_artifacts = save_minmax_cv_artifacts


__all__ = [
    "DEFAULT_N_SPLITS",
    "MinMaxCVResult",
    "TrialGroupedFoldPlan",
    "aggregate_oof_trial_probabilities",
    "aggregate_trial_probabilities",
    "build_minmax_pipelines",
    "build_model_comparison",
    "build_trial_grouped_folds",
    "calculate_fold_metrics",
    "calculate_pooled_metrics",
    "calculate_trial_per_class_metrics",
    "compare_with_subjectwise",
    "load_and_validate_feature_table",
    "make_trial_grouped_folds",
    "run_cross_validation",
    "run_minmax_trial_grouped_cv",
    "save_artifacts",
    "save_minmax_cv_artifacts",
    "validate_window_feature_table",
]
