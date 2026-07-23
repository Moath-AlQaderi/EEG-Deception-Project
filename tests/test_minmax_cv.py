"""Contracts for the separate MinMax trial-grouped cross-validation experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from src.minmax_cv import (
    aggregate_oof_trial_probabilities,
    build_minmax_pipelines,
    build_model_comparison,
    calculate_fold_metrics,
    calculate_pooled_metrics,
    compare_with_subjectwise,
    make_trial_grouped_folds,
    validate_window_feature_table,
)
from src.utils import CLASS_NAMES, MODEL_SLUGS, PROBABILITY_COLUMNS


def _synthetic_windows() -> pd.DataFrame:
    rows = []
    source_index = 0
    for role_internal in range(4):
        # Plenty of groups/class lets StratifiedGroupKFold represent all four
        # classes in every fold; the splitter does not mathematically guarantee
        # this at the minimum of exactly ten groups/class.
        for trial_number in range(50):
            trial_id = f"trial_R{role_internal}_T{trial_number:02d}"
            participant = f"P{trial_number % 5:02d}"
            for window_index in range(2):
                rows.append(
                    {
                        "window_id": f"{trial_id}_W{window_index:02d}",
                        "trial_id": trial_id,
                        "session_id": f"Session{trial_number % 3 + 1}",
                        "participant_id": participant,
                        "question_id": f"Q{trial_number + 1:02d}",
                        "window_index": window_index,
                        "role_id": role_internal + 1,
                        "role_name": CLASS_NAMES[role_internal],
                        "split": "ignored_previous_split",
                        "f__C3__mean": float(source_index),
                        "f__C3__variance": float(source_index % 7),
                    }
                )
                source_index += 1
    return pd.DataFrame(rows)


def test_trial_grouped_folds_cover_rows_once_without_trial_overlap() -> None:
    table = _synthetic_windows()
    names, summary = validate_window_feature_table(table)
    plan = make_trial_grouped_folds(table)
    assert names == ["f__C3__mean", "f__C3__variance"]
    assert summary["prior_split_column_ignored"] is True
    assert len(plan.splits) == 10
    assert plan.assignments["row_position"].is_unique
    assert set(plan.assignments["row_position"]) == set(range(len(table)))
    assert plan.assignments.groupby("trial_id")["fold"].nunique().eq(1).all()
    assert plan.participant_overlap["trial_overlap_count"].eq(0).all()
    assert plan.participant_overlap["all_four_classes_in_test"].all()


def test_all_eight_models_use_fold_local_minmax_pipeline() -> None:
    pipelines = build_minmax_pipelines()
    assert list(pipelines) == MODEL_SLUGS
    for pipeline in pipelines.values():
        assert list(pipeline.named_steps) == ["imputer", "scaler", "classifier"]
        assert isinstance(pipeline.named_steps["scaler"], MinMaxScaler)
        assert pipeline.named_steps["scaler"].feature_range == (0.0, 1.0)

    table = _synthetic_windows()
    plan = make_trial_grouped_folds(table)
    train, test = plan.splits[0]
    X = table[["f__C3__mean", "f__C3__variance"]]
    y = table["role_id"].to_numpy(int) - 1
    model = clone(pipelines["naive_bayes"]).fit(X.iloc[train], y[train])
    assert model.named_steps["scaler"].n_samples_seen_ == len(train)
    assert np.allclose(
        model.named_steps["scaler"].data_min_, X.iloc[train].min().to_numpy()
    )
    assert not np.array_equal(
        model.named_steps["scaler"].data_min_, X.iloc[test].min().to_numpy()
    )


def test_trial_aggregation_uses_mean_probabilities_and_internal_labels() -> None:
    rows = []
    for index, probability in enumerate(
        ([0.1, 0.6, 0.2, 0.1], [0.2, 0.2, 0.5, 0.1])
    ):
        row = {
            "model": "logistic_regression",
            "fold": 3,
            "window_id": f"trial_W{index}",
            "trial_id": "trial",
            "session_id": "Session1",
            "participant_id": "P01",
            "question_id": "Q01",
            "source_role_id": 2,
            "true_label": 1,
            "true_role": "Bluffer",
        }
        row.update(dict(zip(PROBABILITY_COLUMNS, probability)))
        rows.append(row)
    trial = aggregate_oof_trial_probabilities(pd.DataFrame(rows)).iloc[0]
    assert trial["fold"] == 3
    assert trial["true_label"] == 1
    assert trial["predicted_label"] == 1
    assert trial["predicted_role"] == "Bluffer"
    assert np.allclose(
        trial[PROBABILITY_COLUMNS].to_numpy(float), [0.15, 0.4, 0.35, 0.1]
    )


def test_previous_comparison_uses_pooled_trial_columns() -> None:
    old = pd.DataFrame(
        {
            "model": MODEL_SLUGS,
            "trial_accuracy": np.linspace(0.1, 0.2, len(MODEL_SLUGS)),
            "trial_macro_f1": np.linspace(0.2, 0.3, len(MODEL_SLUGS)),
            "trial_cohen_kappa": np.linspace(-0.1, 0.1, len(MODEL_SLUGS)),
        }
    )
    new = pd.DataFrame(
        {
            "model": MODEL_SLUGS,
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "cohen_kappa": 0.3,
        }
    )
    comparison = compare_with_subjectwise(old, new)
    assert list(comparison.columns) == [
        "model",
        "previous_subjectwise_trial_accuracy",
        "previous_subjectwise_macro_f1",
        "previous_subjectwise_kappa",
        "new_non_subjectwise_10fold_trial_accuracy",
        "new_non_subjectwise_macro_f1",
        "new_non_subjectwise_kappa",
    ]
    assert comparison["new_non_subjectwise_macro_f1"].eq(0.4).all()


def test_fold_and_pooled_tables_have_eight_by_ten_coverage() -> None:
    table = _synthetic_windows().reset_index(drop=True)
    plan = make_trial_grouped_folds(table)
    fold_by_row = plan.assignments.set_index("row_position")["fold"]
    rows = []
    for slug in MODEL_SLUGS:
        for index, source in table.iterrows():
            true = int(source["role_id"]) - 1
            probability = np.full(4, 0.1, dtype=float)
            probability[true] = 0.7
            row = {
                "model": slug,
                "fold": int(fold_by_row.loc[index]),
                "source_row_index": index,
                "window_id": source["window_id"],
                "trial_id": source["trial_id"],
                "session_id": source["session_id"],
                "participant_id": source["participant_id"],
                "question_id": source["question_id"],
                "source_role_id": int(source["role_id"]),
                "true_label": true,
                "true_role": CLASS_NAMES[true],
                "predicted_label": true,
                "predicted_role": CLASS_NAMES[true],
            }
            row.update(dict(zip(PROBABILITY_COLUMNS, probability)))
            rows.append(row)
    windows = pd.DataFrame(rows)
    trials = aggregate_oof_trial_probabilities(windows)
    fold_metrics = calculate_fold_metrics(windows, trials)
    pooled_window = calculate_pooled_metrics(windows)
    pooled_trial = calculate_pooled_metrics(trials)
    comparison = build_model_comparison(
        pooled_window, pooled_trial, fold_metrics
    )
    assert len(fold_metrics) == 80
    assert fold_metrics.groupby("model")["fold"].nunique().eq(10).all()
    assert len(pooled_window) == len(pooled_trial) == len(comparison) == 8
    assert comparison["pooled_trial_macro_f1"].eq(1.0).all()


def test_infinite_feature_is_rejected_but_missing_is_imputable() -> None:
    table = _synthetic_windows()
    table.loc[0, "f__C3__mean"] = np.nan
    _, summary = validate_window_feature_table(table)
    assert summary["missing_value_count"] == 1
    table.loc[1, "f__C3__mean"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_window_feature_table(table)
