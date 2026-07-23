"""Vectorized per-channel time and Welch spectral features for EEG windows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import iqr, kurtosis, skew

from .utils import EXPECTED_CHANNELS, FEATURE_PREFIX, stable_json_hash, write_json

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12

TIME_FEATURES = [
    "mean",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "minimum",
    "maximum",
    "peak_to_peak",
    "median",
    "interquartile_range",
    "skewness",
    "kurtosis",
    "mean_absolute_value",
    "waveform_length",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
]

FREQUENCY_FEATURES = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "total_power_1_30_hz",
    "relative_delta_power",
    "relative_theta_power",
    "relative_alpha_power",
    "relative_beta_power",
    "theta_to_alpha_ratio",
    "theta_to_beta_ratio",
    "spectral_entropy",
    "dominant_frequency",
    "median_frequency",
]

ALL_CHANNEL_FEATURES = TIME_FEATURES + FREQUENCY_FEATURES


def ordered_feature_names(channels: Iterable[str] = EXPECTED_CHANNELS) -> list[str]:
    """Return the only ordered column list permitted as model input."""

    return [
        f"{FEATURE_PREFIX}{channel}__{feature}"
        for channel in channels
        for feature in ALL_CHANNEL_FEATURES
    ]


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > EPSILON)
    return result


def _band_power(
    psd: np.ndarray, frequencies: np.ndarray, low: float, high: float, include_high: bool = False
) -> np.ndarray:
    mask = (frequencies >= low) & (
        frequencies <= high if include_high else frequencies < high
    )
    if not np.any(mask):
        return np.full(psd.shape[:-1], np.nan, dtype=np.float64)
    frequency_resolution = float(frequencies[1] - frequencies[0])
    return psd[..., mask].sum(axis=-1) * frequency_resolution


def extract_feature_matrix(
    windows: np.ndarray,
    sampling_rate_hz: int = 256,
    chunk_size: int = 512,
) -> np.ndarray:
    """Extract 31 features/channel from `[windows, 16, 256]` in chunks."""

    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[1:] != (len(EXPECTED_CHANNELS), 256):
        raise ValueError(f"Expected windows [N,16,256], received {windows.shape}")
    feature_count = len(EXPECTED_CHANNELS) * len(ALL_CHANNEL_FEATURES)
    result = np.empty((len(windows), feature_count), dtype=np.float32)

    for start in range(0, len(windows), chunk_size):
        stop = min(start + chunk_size, len(windows))
        signal = np.asarray(windows[start:stop], dtype=np.float64)
        mean = signal.mean(axis=-1)
        variance = signal.var(axis=-1)
        standard_deviation = np.sqrt(variance)
        first_difference = np.diff(signal, axis=-1)
        second_difference = np.diff(first_difference, axis=-1)
        difference_variance = first_difference.var(axis=-1)
        second_difference_variance = second_difference.var(axis=-1)
        mobility = np.sqrt(_safe_ratio(difference_variance, variance))
        derivative_mobility = np.sqrt(
            _safe_ratio(second_difference_variance, difference_variance)
        )
        complexity = _safe_ratio(derivative_mobility, mobility)
        demeaned = signal - mean[..., None]
        zero_crossing_rate = np.mean(
            np.signbit(demeaned[..., 1:]) != np.signbit(demeaned[..., :-1]), axis=-1
        )

        frequencies, psd = welch(
            signal,
            fs=sampling_rate_hz,
            nperseg=128,
            noverlap=64,
            nfft=256,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        delta = _band_power(psd, frequencies, 1.0, 4.0)
        theta = _band_power(psd, frequencies, 4.0, 8.0)
        alpha = _band_power(psd, frequencies, 8.0, 13.0)
        beta = _band_power(psd, frequencies, 13.0, 30.0, include_high=True)
        total = _band_power(psd, frequencies, 1.0, 30.0, include_high=True)
        spectrum_mask = (frequencies >= 1.0) & (frequencies <= 30.0)
        spectrum = psd[..., spectrum_mask]
        spectrum_frequencies = frequencies[spectrum_mask]
        spectrum_sum = spectrum.sum(axis=-1)
        distribution = _safe_ratio(spectrum, spectrum_sum[..., None])
        entropy_terms = np.zeros_like(distribution)
        positive = distribution > 0
        entropy_terms[positive] = distribution[positive] * np.log(distribution[positive])
        entropy_numerator = -np.nansum(entropy_terms, axis=-1)
        spectral_entropy = entropy_numerator / np.log(spectrum.shape[-1])
        spectral_entropy = np.where(spectrum_sum > EPSILON, spectral_entropy, np.nan)
        dominant_frequency = spectrum_frequencies[np.argmax(spectrum, axis=-1)]
        cumulative = np.cumsum(spectrum, axis=-1)
        median_indices = np.argmax(cumulative >= (0.5 * spectrum_sum[..., None]), axis=-1)
        median_frequency = spectrum_frequencies[median_indices]
        dominant_frequency = np.where(spectrum_sum > EPSILON, dominant_frequency, np.nan)
        median_frequency = np.where(spectrum_sum > EPSILON, median_frequency, np.nan)

        features = [
            mean,
            standard_deviation,
            variance,
            np.sqrt(np.mean(np.square(signal), axis=-1)),
            signal.min(axis=-1),
            signal.max(axis=-1),
            np.ptp(signal, axis=-1),
            np.median(signal, axis=-1),
            iqr(signal, axis=-1),
            skew(signal, axis=-1, bias=False),
            kurtosis(signal, axis=-1, fisher=True, bias=False),
            np.mean(np.abs(signal), axis=-1),
            np.sum(np.abs(first_difference), axis=-1),
            zero_crossing_rate,
            variance,
            mobility,
            complexity,
            delta,
            theta,
            alpha,
            beta,
            total,
            _safe_ratio(delta, total),
            _safe_ratio(theta, total),
            _safe_ratio(alpha, total),
            _safe_ratio(beta, total),
            _safe_ratio(theta, alpha),
            _safe_ratio(theta, beta),
            spectral_entropy,
            dominant_frequency,
            median_frequency,
        ]
        stacked = np.stack(features, axis=-1)
        flattened = stacked.reshape(len(signal), feature_count)
        flattened[~np.isfinite(flattened)] = np.nan
        result[start:stop] = flattened.astype(np.float32)
        LOGGER.info("Extracted features for windows %d-%d/%d", start + 1, stop, len(windows))
    return result


def create_feature_table(
    windows: np.ndarray,
    window_metadata: pd.DataFrame,
    outputs_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Combine explicit window metadata with the ordered signal-only feature matrix."""

    if len(windows) != len(window_metadata):
        raise AssertionError("Window array and metadata row counts differ")
    names = ordered_feature_names()
    matrix = extract_feature_matrix(windows)
    if matrix.shape != (len(window_metadata), len(names)):
        raise AssertionError(f"Feature matrix shape mismatch: {matrix.shape}")
    feature_frame = pd.DataFrame(matrix, columns=names)
    table = pd.concat([window_metadata.reset_index(drop=True), feature_frame], axis=1)
    if table["window_id"].duplicated().any():
        raise AssertionError("Duplicate window IDs")
    numeric = table[names].to_numpy()
    if np.isinf(numeric).any():
        raise AssertionError("Infinite feature values remain")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outputs_dir / "window_features.csv", index=False)
    write_json(
        outputs_dir / "feature_schema.json",
        {
            "feature_prefix": FEATURE_PREFIX,
            "ordered_feature_names": names,
            "feature_count": len(names),
            "features_per_channel": len(ALL_CHANNEL_FEATURES),
            "channel_order": EXPECTED_CHANNELS,
            "time_features": TIME_FEATURES,
            "frequency_features": FREQUENCY_FEATURES,
            "welch": {
                "sampling_rate_hz": 256,
                "nperseg": 128,
                "noverlap": 64,
                "nfft": 256,
                "bands_hz": {
                    "delta": "[1,4)",
                    "theta": "[4,8)",
                    "alpha": "[8,13)",
                    "beta": "[13,30]",
                    "total": "[1,30]",
                },
                "absolute_power_scale": "linear microvolt_squared",
                "spectral_entropy_normalized": True,
            },
            "feature_order_sha256": stable_json_hash(names),
            "model_feature_selection_rule": f"exact ordered columns beginning {FEATURE_PREFIX!r}; metadata are never inferred by exclusion",
            "missing_values": int(table[names].isna().sum().sum()),
            "imputation_scope": "not performed here; each fitted model pipeline learns medians from training windows only",
        },
    )
    return table, names
