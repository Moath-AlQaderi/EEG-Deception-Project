"""Preprocess response-free trials, create windows, and extract 496 features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    assert_leakage_free_split,
    assert_no_window_leakage,
    save_metadata_table,
)
from src.feature_extraction import create_feature_table  # noqa: E402
from src.preprocessing import PreprocessingConfig, preprocess_dataset  # noqa: E402
from src.utils import CLASS_NAMES, configure_logging, write_json  # noqa: E402
from src.windowing import WindowConfig, create_window_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_feature_stage(
    metadata: pd.DataFrame,
    project_root: Path,
    outputs_dir: Path,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Run deterministic preprocessing/windowing/features after the frozen split."""

    assert_leakage_free_split(metadata)
    updated, preprocessing_manifest = preprocess_dataset(
        metadata,
        project_root,
        cache_dir,
        outputs_dir,
        PreprocessingConfig(),
    )
    epochs = np.load(cache_dir / "epochs_unnormalized.npy", mmap_mode="r", allow_pickle=False)
    trial_index = pd.read_csv(cache_dir / "trial_index.csv")
    windows, window_index, windowing_manifest = create_window_cache(
        epochs,
        trial_index,
        cache_dir,
        project_root,
        WindowConfig(),
    )
    assert_no_window_leakage(window_index)
    feature_table, feature_names = create_feature_table(
        windows, window_index, outputs_dir
    )

    retained_roles = {
        split: set(group["role_id"].astype(int))
        for split, group in feature_table.groupby("split")
    }
    if retained_roles != {"train": {1, 2, 3, 4}, "test": {1, 2, 3, 4}}:
        raise AssertionError(f"A retained split is missing classes: {retained_roles}")

    window_counts = window_index.groupby("trial_id").size()
    updated["preprocessing_included"] = updated["preprocessing_included"].astype(bool)
    updated["qc_status"] = np.where(
        updated["preprocessing_included"], "included", "excluded"
    )
    updated["rejection_reason"] = updated["exclusion_reason"].fillna("")
    updated["n_windows"] = updated["trial_id"].map(window_counts).fillna(0).astype(int)
    updated["group_random_state"] = 42
    updated["group_test_size"] = 0.20
    save_metadata_table(updated, outputs_dir / "dataset_metadata.csv")
    split_columns = [
        "session_id",
        "participant_id",
        "role_id",
        "question_id",
        "split",
        "trial_id",
        "preprocessing_included",
        "exclusion_reason",
        "n_windows",
        "group_random_state",
        "group_test_size",
    ]
    updated[split_columns].to_csv(outputs_dir / "data_split.csv", index=False)

    train_windows = window_index[window_index["split"] == "train"]
    test_windows = window_index[window_index["split"] == "test"]
    train_participants = set(train_windows["participant_id"])
    test_participants = set(test_windows["participant_id"])
    window_checks = {
        "passed": True,
        "windows_per_retained_trial_exactly_nine": bool(
            window_counts.eq(9).all()
        ),
        "window_ids_unique": bool(window_index["window_id"].is_unique),
        "window_trial_split_unique": bool(
            window_index.groupby("trial_id")["split"].nunique().eq(1).all()
        ),
        "window_participants_disjoint": train_participants.isdisjoint(test_participants),
        "all_four_classes_in_retained_train_test": retained_roles
        == {"train": {1, 2, 3, 4}, "test": {1, 2, 3, 4}},
        "train_windows": int(len(train_windows)),
        "test_windows": int(len(test_windows)),
    }
    if not all(
        value for key, value in window_checks.items() if key not in {"train_windows", "test_windows"}
    ):
        raise AssertionError(f"Window leakage checks failed: {window_checks}")
    leakage = assert_leakage_free_split(updated)
    leakage["window_checks"] = window_checks
    leakage["passed"] = bool(leakage["passed"] and window_checks["passed"])
    write_json(outputs_dir / "leakage_checks.json", leakage)

    manifest = {
        "preprocessing": preprocessing_manifest,
        "windowing": windowing_manifest,
        "feature_count": len(feature_names),
        "feature_rows": len(feature_table),
        "class_order": CLASS_NAMES,
        "retained_trial_counts": {
            str(name): int(count)
            for name, count in trial_index["split"].value_counts().items()
        },
        "window_counts": {
            str(name): int(count)
            for name, count in window_index["split"].value_counts().items()
        },
    }
    write_json(cache_dir / "feature_stage_manifest.json", manifest)
    return updated, feature_table, feature_names, manifest


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("The published experiment seed is fixed at 42")
    outputs = PROJECT_ROOT / "outputs"
    cache = PROJECT_ROOT / "cache"
    configure_logging(outputs, args.verbose)
    metadata_path = outputs / "dataset_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError("Run scripts/inspect_dataset.py first")
    metadata = pd.read_csv(metadata_path)
    updated, features, names, manifest = run_feature_stage(
        metadata, PROJECT_ROOT, outputs, cache
    )
    print(
        "FEATURE SUMMARY\n"
        f"Retained trials: {int(updated['preprocessing_included'].sum())}; "
        f"rejected: {int((~updated['preprocessing_included']).sum())}\n"
        f"Windows: {len(features)}; features/window: {len(names)}\n"
        f"Feature table: {outputs / 'window_features.csv'}"
    )


if __name__ == "__main__":
    main()
