"""Fast contract tests for the fixed classical EEG experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    EXPECTED_TEST_SESSIONS,
    EXPECTED_TRAIN_SESSIONS,
    assign_group_split,
    assert_no_window_leakage,
)
from src.evaluation import aggregate_window_probabilities
from src.feature_extraction import extract_feature_matrix, ordered_feature_names
from src.models import SCALED_MODELS, build_model_pipelines, fit_model_pipelines
from src.preprocessing import (
    PreprocessingConfig,
    _filter_and_resample_pre_response,
    _phase_slice,
    _raw_qc,
    _resampled_index,
)
from src.utils import EXPECTED_CHANNELS, MODEL_SLUGS, PROBABILITY_COLUMNS, write_json
from src.windowing import WindowConfig, window_epochs


def _complete_raw_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    participant_number = 1
    for session_number in range(1, 18):
        session = f"Session{session_number}"
        honest = f"P{participant_number:02d}"
        bluffer = f"P{participant_number + 1:02d}"
        thief = f"P{participant_number + 2:02d}"
        participant_number += 3
        for role_id, participant in ((1, honest), (2, bluffer), (3, thief), (4, thief)):
            for question_number in range(1, 15):
                question = f"Q{question_number:02d}"
                rows.append(
                    {
                        "trial_id": f"{session}_{participant}_R{role_id:02d}_{question}",
                        "session_id": session,
                        "participant_id": participant,
                        "role_id": role_id,
                        "question_id": question,
                    }
                )
    return pd.DataFrame(rows)


def test_frozen_string_group_shuffle_split_and_leakage_contract() -> None:
    assigned, summary = assign_group_split(_complete_raw_metadata())
    train = assigned[assigned["split"] == "train"]
    test = assigned[assigned["split"] == "test"]
    assert set(train["session_id"]) == EXPECTED_TRAIN_SESSIONS
    assert set(test["session_id"]) == EXPECTED_TEST_SESSIONS
    assert len(train) == 728 and len(test) == 224
    assert train["participant_id"].nunique() == 39
    assert test["participant_id"].nunique() == 12
    assert set(train["participant_id"]).isdisjoint(test["participant_id"])
    assert summary["leakage_checks"]["passed"] is True
    with pytest.raises(ValueError, match="frozen"):
        assign_group_split(_complete_raw_metadata(), random_state=7)


def test_one_based_inclusive_event_slicing_and_resampled_start() -> None:
    signal = np.arange(20, dtype=float)[None, :]
    events = {"rest_start": 2, "rest_end": 4}
    assert np.array_equal(_phase_slice(signal, events, "rest"), [[1.0, 2.0, 3.0]])
    assert _resampled_index(1, 2048, 256) == 0
    assert _resampled_index(9, 2048, 256) == 1
    assert _resampled_index(10, 2048, 256) == 2


def test_steady_state_filter_has_no_constant_signal_startup_and_qc_is_continuous() -> None:
    config = PreprocessingConfig()
    constants = np.linspace(-0.02, 0.02, 16)[:, None]
    pre_response = np.repeat(constants, 4096, axis=1)
    filtered = _filter_and_resample_pre_response(pre_response, 2048, config)
    assert filtered.shape == (16, 512)
    assert np.max(np.abs(filtered)) < 1e-9

    contaminated = np.zeros((16, 4096), dtype=float)
    contaminated[3, 2048:] = 0.1
    reasons, metrics = _raw_qc(contaminated, {}, config)
    assert "large_raw_jump" in reasons
    assert metrics["raw_jump_channel_count"] == 1


def test_windowing_yields_nine_aligned_one_second_windows() -> None:
    epochs = np.arange(2 * 16 * 1280, dtype=np.float32).reshape(2, 16, 1280)
    trial_index = pd.DataFrame(
        {
            "array_index": [0, 1],
            "trial_id": ["trial_train", "trial_test"],
            "session_id": ["Session2", "Session1"],
            "participant_id": ["P04", "P01"],
            "role_id": [1, 2],
            "question_id": ["Q01", "Q01"],
            "split": ["train", "test"],
        }
    )
    windows, index = window_epochs(epochs, trial_index, WindowConfig())
    assert windows.shape == (18, 16, 256)
    assert index.groupby("trial_id").size().to_dict() == {
        "trial_test": 9,
        "trial_train": 9,
    }
    assert index[index["trial_id"] == "trial_train"]["window_start_sample"].tolist() == list(
        range(0, 1025, 128)
    )
    assert np.array_equal(windows[0], epochs[0, :, :256])
    assert np.array_equal(windows[8], epochs[0, :, 1024:1280])
    assert_no_window_leakage(index)


def test_feature_extraction_has_exact_signal_only_shape_and_no_infinity() -> None:
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(2, len(EXPECTED_CHANNELS), 256)).astype(np.float32)
    matrix = extract_feature_matrix(windows, chunk_size=1)
    names = ordered_feature_names()
    assert matrix.shape == (2, 496)
    assert len(names) == len(set(names)) == 496
    assert all(name.startswith("f__") for name in names)
    assert not np.isinf(matrix).any()
    assert np.isfinite(matrix).all()


def test_zero_energy_features_become_nan_not_infinity() -> None:
    zeros = np.zeros((1, len(EXPECTED_CHANNELS), 256), dtype=np.float32)
    with np.errstate(all="ignore"):
        matrix = extract_feature_matrix(zeros)
    assert not np.isinf(matrix).any()
    assert np.isnan(matrix).any()


def test_fixed_model_parameters_and_training_only_pipeline_steps() -> None:
    models = build_model_pipelines(seed=42)
    assert list(models) == MODEL_SLUGS
    for slug, pipeline in models.items():
        expected = ["imputer"] + (["scaler"] if slug in SCALED_MODELS else []) + [
            "classifier"
        ]
        assert list(pipeline.named_steps) == expected
        assert pipeline.named_steps["imputer"].strategy == "median"
    assert models["random_forest"].named_steps["classifier"].n_estimators == 200
    assert models["svm"].named_steps["classifier"].probability is True
    assert models["mlp"].named_steps["classifier"].hidden_layer_sizes == (128, 64)
    assert models["xgboost"].named_steps["classifier"].objective == "multi:softprob"


def test_metadata_columns_are_rejected_from_model_inputs_before_fit() -> None:
    bad = pd.DataFrame({"f__C3__mean": [0.0, 1.0, 2.0, 3.0], "participant_id": [1, 2, 3, 4]})
    with pytest.raises(AssertionError, match="Metadata-like"):
        fit_model_pipelines(bad, np.arange(4), feature_names=list(bad.columns))


def test_trial_prediction_is_mean_of_window_probabilities() -> None:
    rows = []
    probabilities = ([0.7, 0.1, 0.1, 0.1], [0.1, 0.5, 0.2, 0.2])
    for index, probability in enumerate(probabilities):
        row = {
            "model": "logistic_regression",
            "window_id": f"trial_1_W{index + 1:02d}",
            "trial_id": "trial_1",
            "session_id": "Session1",
            "participant_id": "P01",
            "question_id": "Q01",
            "role_id": 1,
            "role_name": "Honest",
            "split": "test",
            "true_label": 1,
            "true_role": "Honest",
        }
        row.update(dict(zip(PROBABILITY_COLUMNS, probability)))
        rows.append(row)
    trial = aggregate_window_probabilities(pd.DataFrame(rows)).iloc[0]
    assert trial["n_windows"] == 2
    assert np.allclose(trial[PROBABILITY_COLUMNS].to_numpy(float), [0.4, 0.3, 0.15, 0.15])
    assert trial["predicted_label"] == 1


def test_probability_argmax_and_strict_json_edge_cases(tmp_path) -> None:
    near_tie = pd.DataFrame(
        [
            {
                "model": "svm",
                "window_id": "trial_2_W01",
                "trial_id": "trial_2",
                "session_id": "Session1",
                "participant_id": "P01",
                "question_id": "Q02",
                "true_label": 2,
                **dict(
                    zip(
                        PROBABILITY_COLUMNS,
                        [0.2, 0.3, 0.2, 0.30000000000000004],
                    )
                ),
            }
        ]
    )
    trial = aggregate_window_probabilities(near_tie).iloc[0]
    assert trial["predicted_label"] == 4

    destination = tmp_path / "strict.json"
    write_json(
        destination,
        {"finite": 1.0, "nan": float("nan"), "inf": np.inf, "numpy_bool": np.bool_(True)},
    )
    text = destination.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert '"nan": null' in text and '"inf": null' in text
    assert '"numpy_bool": true' in text
