"""Deterministic 1-second, 50%-overlap windows for cached EEG epochs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import portable_path, stable_json_hash, write_json


@dataclass(frozen=True)
class WindowConfig:
    sampling_rate_hz: int = 256
    epoch_samples: int = 1280
    window_samples: int = 256
    hop_samples: int = 128
    expected_windows_per_trial: int = 9

    def validate(self) -> None:
        if min(
            self.sampling_rate_hz,
            self.epoch_samples,
            self.window_samples,
            self.hop_samples,
        ) <= 0:
            raise ValueError("Window dimensions and sampling rate must be positive")
        if self.window_samples > self.epoch_samples:
            raise ValueError("Window cannot be longer than the epoch")
        starts = self.starts
        if len(starts) != self.expected_windows_per_trial:
            raise ValueError(
                f"Configuration yields {len(starts)} windows, expected "
                f"{self.expected_windows_per_trial}"
            )
        if self.window_samples != self.sampling_rate_hz:
            raise ValueError("The predefined experiment requires one-second windows")
        if self.hop_samples * 2 != self.window_samples:
            raise ValueError("The predefined experiment requires 50% overlap")

    @property
    def starts(self) -> tuple[int, ...]:
        return tuple(
            range(0, self.epoch_samples - self.window_samples + 1, self.hop_samples)
        )


def window_epochs(
    epochs: np.ndarray,
    trial_index: pd.DataFrame,
    config: WindowConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return flattened windows and one aligned metadata row per window."""

    config = config or WindowConfig()
    config.validate()
    array = np.asarray(epochs)
    if array.ndim != 3:
        raise ValueError("Epochs must have shape trials x channels x samples")
    if array.shape[-1] != config.epoch_samples:
        raise ValueError(
            f"Expected {config.epoch_samples} epoch samples, received {array.shape[-1]}"
        )
    if len(trial_index) != len(array):
        raise ValueError("Trial index and epoch cache have different row counts")
    required = {
        "array_index",
        "trial_id",
        "session_id",
        "participant_id",
        "role_id",
        "question_id",
        "split",
    }
    missing = sorted(required.difference(trial_index.columns))
    if missing:
        raise ValueError(f"Trial index is missing columns: {missing}")
    if not np.array_equal(trial_index["array_index"], np.arange(len(trial_index))):
        raise AssertionError("Trial index is not aligned with epoch rows")
    if trial_index["trial_id"].duplicated().any():
        raise AssertionError("Trial IDs must be unique before window expansion")
    if not np.isfinite(array).all():
        raise ValueError("Epoch cache contains non-finite values")

    starts = config.starts
    # Shape before flattening: trials x windows x channels x samples.
    four_dimensional = np.stack(
        [array[:, :, start : start + config.window_samples] for start in starts],
        axis=1,
    )
    windows = four_dimensional.reshape(
        len(array) * len(starts), array.shape[1], config.window_samples
    ).astype(np.float32, copy=False)

    repeated = trial_index.loc[trial_index.index.repeat(len(starts))].reset_index(
        drop=True
    )
    repeated["window_index"] = np.tile(np.arange(len(starts)), len(array))
    repeated["window_id"] = [
        f"{trial_id}_W{int(window_index) + 1:02d}"
        for trial_id, window_index in zip(
            repeated["trial_id"], repeated["window_index"]
        )
    ]
    repeated["window_start_sample"] = np.tile(starts, len(array))
    repeated["window_stop_sample_exclusive"] = (
        repeated["window_start_sample"] + config.window_samples
    )
    repeated["window_start_seconds"] = (
        repeated["window_start_sample"] / config.sampling_rate_hz
    )
    repeated["window_stop_seconds"] = (
        repeated["window_stop_sample_exclusive"] / config.sampling_rate_hz
    )
    repeated.insert(0, "window_array_index", np.arange(len(repeated)))

    counts = repeated.groupby("trial_id", sort=False)["window_index"].nunique()
    if not counts.eq(config.expected_windows_per_trial).all():
        raise AssertionError("Every retained trial must yield exactly nine windows")
    if repeated.groupby("trial_id")["split"].nunique().gt(1).any():
        raise AssertionError("Windows from a trial crossed the fixed data split")
    if repeated["window_id"].duplicated().any():
        raise AssertionError("Window IDs must be globally unique")
    if windows.shape != (
        len(array) * config.expected_windows_per_trial,
        array.shape[1],
        config.window_samples,
    ):
        raise AssertionError(f"Unexpected window tensor shape: {windows.shape}")
    return windows, repeated


def create_window_cache(
    epochs: np.ndarray,
    trial_index: pd.DataFrame,
    cache_dir: Path,
    project_root: Path,
    config: WindowConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Create and save the deterministic response-free window cache."""

    config = config or WindowConfig()
    windows, window_index = window_epochs(epochs, trial_index, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows_path = cache_dir / "windows.npy"
    index_path = cache_dir / "window_index.csv"
    np.save(windows_path, windows, allow_pickle=False)
    window_index.to_csv(index_path, index=False)
    manifest: dict[str, Any] = {
        "config": asdict(config),
        "config_sha256": stable_json_hash(config),
        "window_shape": list(windows.shape),
        "window_dtype": str(windows.dtype),
        "trial_count": int(len(trial_index)),
        "window_count": int(len(window_index)),
        "windows_per_trial": config.expected_windows_per_trial,
        "window_seconds": config.window_samples / config.sampling_rate_hz,
        "hop_seconds": config.hop_samples / config.sampling_rate_hz,
        "overlap_fraction": 1.0 - config.hop_samples / config.window_samples,
        "split_preservation": "every window inherits its source trial/session split",
        "paths": {
            "windows": portable_path(windows_path, project_root),
            "window_index": portable_path(index_path, project_root),
        },
    }
    write_json(cache_dir / "windowing_manifest.json", manifest)
    return windows, window_index, manifest


def load_window_cache(
    cache_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load and validate cached windows and their aligned metadata."""

    import json

    windows = np.load(cache_dir / "windows.npy", mmap_mode="r", allow_pickle=False)
    index = pd.read_csv(cache_dir / "window_index.csv")
    manifest = json.loads(
        (cache_dir / "windowing_manifest.json").read_text(encoding="utf-8")
    )
    if tuple(windows.shape) != tuple(manifest["window_shape"]):
        raise AssertionError("Window cache does not match its manifest")
    if len(index) != len(windows):
        raise AssertionError("Window cache and index row counts differ")
    if not np.array_equal(index["window_array_index"], np.arange(len(index))):
        raise AssertionError("Window index is not aligned with window rows")
    if index.groupby("trial_id")["window_index"].nunique().ne(9).any():
        raise AssertionError("A cached trial does not have exactly nine windows")
    return windows, index, manifest


__all__ = [
    "WindowConfig",
    "create_window_cache",
    "load_window_cache",
    "window_epochs",
]
