"""Response-free, label-blind preprocessing for the classical EEG pipeline.

The MATLAB event markers are one-based and inclusive. The continuous record is
truncated at ``thinking_end`` before filtering, so overt response samples never
enter filtering, QC, resampling, baseline correction, or a model epoch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from mne.filter import construct_iir_filter
from scipy.io import loadmat
from scipy.signal import resample_poly, sosfilt, sosfilt_zi

from .utils import (
    EXPECTED_CHANNELS,
    portable_path,
    resolve_stored_path,
    stable_json_hash,
    write_json,
)


LOGGER = logging.getLogger(__name__)
EVENT_PHASES = ("rest", "question", "answers", "thinking", "response")
EVENT_FIELDS = tuple(
    f"{phase}_{edge}" for phase in EVENT_PHASES for edge in ("start", "end")
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fixed preprocessing and conservative severe-corruption thresholds."""

    source_sampling_rate_hz: int = 2048
    target_sampling_rate_hz: int = 256
    thinking_seconds: float = 5.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 32.5
    butterworth_order: int = 4
    reference: str = "common_average"
    source_units: str = "volts"
    output_units: str = "microvolts"
    raw_acquisition_rail_v: float = 0.25
    raw_large_jump_v: float = 0.05
    flat_channel_std_uv: float = 0.1
    noisy_channel_std_uv: float = 1000.0
    extreme_channel_peak_to_peak_uv: float = 5000.0

    @property
    def source_epoch_samples(self) -> int:
        return int(round(self.thinking_seconds * self.source_sampling_rate_hz))

    @property
    def output_epoch_samples(self) -> int:
        return int(round(self.thinking_seconds * self.target_sampling_rate_hz))

    def validate(self) -> None:
        if self.source_sampling_rate_hz <= 0 or self.target_sampling_rate_hz <= 0:
            raise ValueError("Sampling rates must be positive")
        if self.source_sampling_rate_hz % self.target_sampling_rate_hz != 0:
            raise ValueError("This pipeline requires an integer downsampling ratio")
        if self.thinking_seconds <= 0:
            raise ValueError("thinking_seconds must be positive")
        if not 0 < self.bandpass_low_hz < self.bandpass_high_hz:
            raise ValueError("Band-pass edges must be positive and ordered")
        if self.bandpass_high_hz >= self.target_sampling_rate_hz / 2:
            raise ValueError("The upper band-pass edge must be below target Nyquist")
        if self.butterworth_order != 4:
            raise ValueError("The predefined experiment requires a fourth-order filter")
        if self.reference != "common_average":
            raise ValueError("The predefined experiment requires common-average reference")
        if self.source_units != "volts" or self.output_units != "microvolts":
            raise ValueError("The predefined unit conversion is volts to microvolts")
        thresholds = (
            self.raw_acquisition_rail_v,
            self.raw_large_jump_v,
            self.flat_channel_std_uv,
            self.noisy_channel_std_uv,
            self.extreme_channel_peak_to_peak_uv,
        )
        if any(value <= 0 for value in thresholds):
            raise ValueError("All QC thresholds must be positive")


def _load_record(path: Path) -> Mapping[str, Any]:
    payload = loadmat(path, simplify_cells=True, variable_names=["EEGStruct"])
    record = payload.get("EEGStruct")
    if not isinstance(record, Mapping):
        raise ValueError("MAT file does not contain a scalar EEGStruct")
    return record


def _event_values(record: Mapping[str, Any]) -> dict[str, int]:
    events = record.get("events")
    if not isinstance(events, Mapping):
        raise ValueError("EEGStruct.events is not a mapping")
    missing = [field for field in EVENT_FIELDS if field not in events]
    if missing:
        raise ValueError(f"Missing event markers: {missing}")
    return {field: int(events[field]) for field in EVENT_FIELDS}


def _validate_record(
    data: np.ndarray,
    channels: list[str],
    sampling_rate: int,
    events: Mapping[str, int],
    config: PreprocessingConfig,
) -> list[str]:
    reasons: list[str] = []
    if data.ndim != 2 or data.shape[0] != len(EXPECTED_CHANNELS):
        return ["incorrect_signal_shape"]
    if channels != EXPECTED_CHANNELS:
        reasons.append("unexpected_channel_order")
    if sampling_rate != config.source_sampling_rate_hz:
        reasons.append("unexpected_sampling_rate")
    for phase in EVENT_PHASES:
        start = events[f"{phase}_start"]
        end = events[f"{phase}_end"]
        if start < 1 or end < start or end > data.shape[1]:
            reasons.append(f"invalid_{phase}_bounds")
    for first, second in zip(EVENT_PHASES[:-1], EVENT_PHASES[1:]):
        if events[f"{second}_start"] != events[f"{first}_end"] + 1:
            reasons.append(f"noncontiguous_{first}_{second}")
    available = events["thinking_end"] - events["thinking_start"] + 1
    if available < config.source_epoch_samples:
        reasons.append("thinking_phase_too_short")
    if events["rest_start"] != 1:
        reasons.append("rest_does_not_start_at_first_sample")
    return list(dict.fromkeys(reasons))


def _phase_slice(data: np.ndarray, events: Mapping[str, int], phase: str) -> np.ndarray:
    """Slice one MATLAB one-based inclusive event interval."""

    return data[:, events[f"{phase}_start"] - 1 : events[f"{phase}_end"]]


def _raw_qc(
    pre_response: np.ndarray,
    events: Mapping[str, int],
    config: PreprocessingConfig,
) -> tuple[list[str], dict[str, float | int]]:
    if not np.isfinite(pre_response).all():
        return ["nonfinite_pre_response"], {
            "raw_rail_channel_count": -1,
            "raw_jump_channel_count": -1,
            "raw_relevant_peak_abs_v": np.nan,
            "raw_pre_response_peak_abs_v": np.nan,
        }

    # Every sample that enters the causal filter is included in raw QC. This
    # catches rails or discontinuities in the question/answer interval that
    # would otherwise propagate forward into thinking.
    relevant = pre_response
    rail_by_channel = (
        np.max(np.abs(relevant), axis=1) >= config.raw_acquisition_rail_v
    )
    jump_by_channel = (
        np.max(np.abs(np.diff(relevant, axis=1)), axis=1)
        >= config.raw_large_jump_v
    )
    reasons: list[str] = []
    if rail_by_channel.any():
        reasons.append("acquisition_rail")
    if jump_by_channel.any():
        reasons.append("large_raw_jump")
    return reasons, {
        "raw_rail_channel_count": int(rail_by_channel.sum()),
        "raw_jump_channel_count": int(jump_by_channel.sum()),
        "raw_relevant_peak_abs_v": float(np.max(np.abs(relevant))),
        "raw_pre_response_peak_abs_v": float(np.max(np.abs(relevant))),
    }


def _resampled_index(
    matlab_one_based_index: int,
    source_rate: int,
    target_rate: int,
) -> int:
    """First target-grid sample at or after a MATLAB event-start sample."""

    source_index0 = matlab_one_based_index - 1
    return int(np.ceil(source_index0 * target_rate / source_rate - 1e-12))


def _filter_and_resample_pre_response(
    pre_response: np.ndarray,
    sampling_rate: int,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Causally filter response-truncated data with steady-state SOS initialization."""

    iir = construct_iir_filter(
        {
            "order": config.butterworth_order,
            "ftype": "butter",
            "output": "sos",
        },
        f_pass=[config.bandpass_low_hz, config.bandpass_high_hz],
        sfreq=sampling_rate,
        btype="bandpass",
        phase="forward",
        return_copy=True,
        verbose=False,
    )
    sos = np.asarray(iir["sos"], dtype=np.float64)
    # A zero-state causal high-pass filter creates a long artificial transient in
    # the 0.5-s rest baseline. Initialize each channel at the steady state of its
    # first observed value instead.
    initial_state = (
        sosfilt_zi(sos)[:, None, :]
        * pre_response[:, 0][None, :, None]
    )
    filtered, _ = sosfilt(sos, pre_response, axis=-1, zi=initial_state)
    divisor = int(np.gcd(config.source_sampling_rate_hz, config.target_sampling_rate_hz))
    return resample_poly(
        filtered,
        up=config.target_sampling_rate_hz // divisor,
        down=config.source_sampling_rate_hz // divisor,
        axis=1,
        padtype="line",
    )


def preprocess_trial(
    path: Path,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray | None, list[str], dict[str, float | int]]:
    """Return one response-free ``[16, 1280]`` float32 microvolt epoch."""

    config = config or PreprocessingConfig()
    config.validate()
    record = _load_record(path)
    required = {"data", "events", "channels", "sampling_rate"}
    missing = sorted(required.difference(record))
    if missing:
        return None, [f"missing_fields:{','.join(missing)}"], {}

    data = np.asarray(record["data"], dtype=np.float64)
    channels = [str(value) for value in np.asarray(record["channels"]).reshape(-1)]
    sampling_rate = int(record["sampling_rate"])
    events = _event_values(record)
    reasons = _validate_record(data, channels, sampling_rate, events, config)
    if reasons:
        return None, reasons, {}

    # thinking_end is one-based inclusive. The exclusive Python stop is therefore
    # exactly thinking_end, and response_start maps to the next unused sample.
    pre_response = data[:, : events["thinking_end"]]
    raw_reasons, metrics = _raw_qc(pre_response, events, config)
    if raw_reasons:
        return None, raw_reasons, metrics

    # Filtering the response-truncated continuous segment preserves filter state
    # across phase boundaries without ever exposing it to speech/movement data.
    downsampled = _filter_and_resample_pre_response(
        pre_response, sampling_rate, config
    )

    downsampled -= downsampled.mean(axis=0, keepdims=True)
    rest_start = _resampled_index(
        events["rest_start"], sampling_rate, config.target_sampling_rate_hz
    )
    rest_stop = int(
        np.ceil(events["rest_end"] * config.target_sampling_rate_hz / sampling_rate)
    )
    rest = downsampled[:, rest_start:rest_stop]
    if rest.shape[1] == 0:
        return None, ["empty_resampled_rest_phase"], metrics
    baseline = rest.mean(axis=1, keepdims=True)
    thinking_start = _resampled_index(
        events["thinking_start"], sampling_rate, config.target_sampling_rate_hz
    )
    thinking_stop = thinking_start + config.output_epoch_samples
    if thinking_stop > downsampled.shape[1]:
        return None, ["resampled_thinking_phase_too_short"], metrics
    epoch_uv = (
        downsampled[:, thinking_start:thinking_stop] - baseline
    ) * 1_000_000.0
    if epoch_uv.shape != (len(EXPECTED_CHANNELS), config.output_epoch_samples):
        return None, [f"unexpected_epoch_shape:{epoch_uv.shape}"], metrics
    if not np.isfinite(epoch_uv).all():
        return None, ["nonfinite_processed_epoch"], metrics

    channel_std = np.std(epoch_uv, axis=1)
    channel_peak_to_peak = np.ptp(epoch_uv, axis=1)
    flat = channel_std < config.flat_channel_std_uv
    noisy = channel_std > config.noisy_channel_std_uv
    extreme = channel_peak_to_peak > config.extreme_channel_peak_to_peak_uv
    metrics.update(
        {
            "processed_min_channel_std_uv": float(channel_std.min()),
            "processed_max_channel_std_uv": float(channel_std.max()),
            "processed_max_peak_to_peak_uv": float(channel_peak_to_peak.max()),
            "processed_flat_channel_count": int(flat.sum()),
            "processed_noisy_channel_count": int(noisy.sum()),
            "processed_extreme_peak_to_peak_channel_count": int(extreme.sum()),
            "thinking_start_resampled_index": thinking_start,
            "thinking_start_alignment_error_ms": float(
                (
                    thinking_start / config.target_sampling_rate_hz
                    - (events["thinking_start"] - 1) / sampling_rate
                )
                * 1000.0
            ),
        }
    )
    processed_reasons: list[str] = []
    if flat.any():
        processed_reasons.append("flat_processed_channel")
    if noisy.any():
        processed_reasons.append("noisy_processed_channel")
    if extreme.any():
        processed_reasons.append("extreme_processed_peak_to_peak")
    if processed_reasons:
        return None, processed_reasons, metrics
    return np.asarray(epoch_uv, dtype=np.float32), [], metrics


def preprocess_dataset(
    metadata: pd.DataFrame,
    project_root: Path,
    cache_dir: Path,
    outputs_dir: Path,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preprocess all trials and save aligned response-free epochs and QC."""

    config = config or PreprocessingConfig()
    config.validate()
    required = {
        "filepath",
        "trial_id",
        "session_id",
        "participant_id",
        "role_id",
        "question_id",
        "split",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata are missing required columns: {missing}")
    if metadata["trial_id"].duplicated().any():
        raise ValueError("Metadata contain duplicate trial_id values")

    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    updated = metadata.copy()
    updated["preprocessing_included"] = False
    updated["exclusion_reason"] = ""
    epochs: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for position, (row_index, row) in enumerate(updated.iterrows(), start=1):
        reasons: list[str] = []
        metrics: dict[str, Any] = {
            "trial_id": row["trial_id"],
            "session_id": row["session_id"],
            "participant_id": row["participant_id"],
            "role_id": int(row["role_id"]),
            "question_id": row["question_id"],
            "split": row["split"],
            "included": False,
        }
        valid_column = next(
            (
                name
                for name in ("structurally_valid", "structural_valid")
                if name in updated.columns
            ),
            None,
        )
        if valid_column is not None and not bool(row[valid_column]):
            reasons.append("structurally_invalid")
            epoch = None
        else:
            try:
                epoch, reasons, trial_metrics = preprocess_trial(
                    resolve_stored_path(str(row["filepath"]), project_root), config
                )
                metrics.update(trial_metrics)
            except Exception as error:
                epoch = None
                reasons = [f"preprocessing_error:{type(error).__name__}:{error}"]

        if epoch is not None and not reasons:
            array_index = len(epochs)
            epochs.append(epoch)
            metrics["included"] = True
            index_row = {
                "array_index": array_index,
                "trial_id": row["trial_id"],
                "filepath": row["filepath"],
                "session_id": row["session_id"],
                "participant_id": row["participant_id"],
                "role_id": int(row["role_id"]),
                "question_id": row["question_id"],
                "split": row["split"],
            }
            if "role_name" in updated.columns:
                index_row["role_name"] = row["role_name"]
            index_rows.append(index_row)
            updated.at[row_index, "preprocessing_included"] = True
        else:
            updated.at[row_index, "exclusion_reason"] = "|".join(
                dict.fromkeys(reasons)
            )
        metrics["exclusion_reason"] = updated.at[row_index, "exclusion_reason"]
        qc_rows.append(metrics)
        if position % 100 == 0 or position == len(updated):
            LOGGER.info("Preprocessed %d/%d trials", position, len(updated))

    if not epochs:
        raise RuntimeError("All trials were excluded during preprocessing")
    epoch_array = np.stack(epochs).astype(np.float32, copy=False)
    expected = (len(epochs), len(EXPECTED_CHANNELS), config.output_epoch_samples)
    if epoch_array.shape != expected or not np.isfinite(epoch_array).all():
        raise AssertionError(
            f"Invalid epoch cache: observed={epoch_array.shape}, expected={expected}"
        )
    index_table = pd.DataFrame(index_rows)
    if not np.array_equal(index_table["array_index"], np.arange(len(index_table))):
        raise AssertionError("Epoch index is not contiguous")

    epochs_path = cache_dir / "epochs_unnormalized.npy"
    index_path = cache_dir / "trial_index.csv"
    qc_path = outputs_dir / "preprocessing_qc.csv"
    np.save(epochs_path, epoch_array, allow_pickle=False)
    index_table.to_csv(index_path, index=False)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

    exclusion_counts = Counter(
        reason
        for reason in updated.loc[
            ~updated["preprocessing_included"], "exclusion_reason"
        ]
    )
    manifest: dict[str, Any] = {
        "config": asdict(config),
        "config_sha256": stable_json_hash(config),
        "event_index_convention": "MATLAB one-based inclusive",
        "response_samples_used": False,
        "filter_scope": "trial start through inclusive thinking_end; response excluded before filtering",
        "filter": "MNE-designed forward fourth-order Butterworth SOS IIR, 0.5-32.5 Hz, scipy sosfilt with per-channel steady-state initialization",
        "filter_initialization": "sosfilt_zi scaled by each channel's first pre-response sample",
        "resampling": "complete filtered pre-response segment, polyphase 2048 to 256 Hz",
        "question_answer_samples_used": "filter state and QC only; never extracted as analysis samples",
        "raw_qc_scope": "every sample entering the causal filter, from trial start through thinking_end",
        "reference": "common average after resampling",
        "baseline": "same-trial resampled rest mean after common-average reference",
        "units": "explicit volts to microvolts conversion after baseline correction",
        "cache_dtype": str(epoch_array.dtype),
        "epoch_shape": list(epoch_array.shape),
        "included_trial_count": int(len(index_table)),
        "excluded_trial_count": int(len(updated) - len(index_table)),
        "included_by_split": {
            str(name): int(count)
            for name, count in index_table["split"].value_counts().items()
        },
        "exclusion_reason_counts": dict(exclusion_counts),
        "paths": {
            "epochs": portable_path(epochs_path, project_root),
            "trial_index": portable_path(index_path, project_root),
            "qc": portable_path(qc_path, project_root),
        },
    }
    write_json(cache_dir / "preprocessing_manifest.json", manifest)
    return updated, manifest


def load_preprocessed_cache(
    cache_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load and validate the response-free epoch cache."""

    epochs = np.load(
        cache_dir / "epochs_unnormalized.npy", mmap_mode="r", allow_pickle=False
    )
    index_table = pd.read_csv(cache_dir / "trial_index.csv")
    import json

    manifest = json.loads(
        (cache_dir / "preprocessing_manifest.json").read_text(encoding="utf-8")
    )
    expected_shape = tuple(manifest["epoch_shape"])
    if tuple(epochs.shape) != expected_shape or epochs.dtype != np.float32:
        raise AssertionError("Epoch cache does not match its manifest")
    if len(index_table) != len(epochs):
        raise AssertionError("Epoch cache and trial index have different row counts")
    if not np.array_equal(index_table["array_index"], np.arange(len(index_table))):
        raise AssertionError("Trial index is not aligned with the epoch cache")
    if index_table["trial_id"].duplicated().any():
        raise AssertionError("Trial index contains duplicate trial IDs")
    return epochs, index_table, manifest


__all__ = [
    "PreprocessingConfig",
    "load_preprocessed_cache",
    "preprocess_dataset",
    "preprocess_trial",
]
