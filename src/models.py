"""Fixed classical model pipelines for the eight-model EEG benchmark.

There is intentionally no search or cross-validation in this module.  Every
pipeline owns its imputer and, where appropriate, scaler so those transforms are
fitted only when the pipeline is fitted on the training windows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from .utils import CLASS_NAMES, FEATURE_PREFIX, MODEL_SLUGS, RANDOM_SEED


LOGGER = logging.getLogger(__name__)
INTERNAL_CLASS_LABELS = np.arange(len(CLASS_NAMES), dtype=int)
MODEL_FILENAMES = {slug: f"{slug}.joblib" for slug in MODEL_SLUGS}
SCALED_MODELS = {"logistic_regression", "svm", "knn", "naive_bayes", "mlp"}


def _steps(classifier: BaseEstimator, *, scale: bool) -> list[tuple[str, BaseEstimator]]:
    """Create fresh preprocessing steps for one independently fitted model."""

    steps: list[tuple[str, BaseEstimator]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return steps


def build_model_pipelines(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Return the eight fixed, prompt-specified classical pipelines."""

    classifiers: dict[str, BaseEstimator] = {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_NAMES),
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=False,
            random_state=seed,
        ),
    }
    if list(classifiers) != list(MODEL_SLUGS):
        raise AssertionError("Model construction order differs from utils.MODEL_SLUGS")
    return {
        slug: Pipeline(_steps(classifier, scale=slug in SCALED_MODELS))
        for slug, classifier in classifiers.items()
    }


# A concise alias is convenient for scripts while retaining the descriptive API.
create_model_pipelines = build_model_pipelines
build_models = build_model_pipelines


def _validate_training_arrays(
    X: Any,
    y: Sequence[int],
    feature_names: Sequence[str] | None,
) -> tuple[Any, np.ndarray]:
    if feature_names is not None:
        names = [str(name) for name in feature_names]
    elif hasattr(X, "columns"):
        names = [str(name) for name in X.columns]
    else:
        names = []
    if names and any(not name.startswith(FEATURE_PREFIX) for name in names):
        bad = [name for name in names if not name.startswith(FEATURE_PREFIX)]
        raise AssertionError(f"Metadata-like columns cannot enter model X: {bad[:5]}")
    if hasattr(X, "to_numpy"):
        numeric = np.asarray(X.to_numpy(), dtype=float)
    else:
        numeric = np.asarray(X, dtype=float)
    if numeric.ndim != 2 or numeric.shape[0] == 0 or numeric.shape[1] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional feature matrix")
    if np.isinf(numeric).any():
        raise ValueError("X_train contains infinite feature values")
    labels = np.asarray(y, dtype=int)
    if labels.shape != (numeric.shape[0],):
        raise ValueError("y_train must contain one internal label per training window")
    observed = np.unique(labels)
    if not np.array_equal(observed, INTERNAL_CLASS_LABELS):
        raise ValueError(
            "Training labels must use all four internal classes 0..3; "
            f"observed {observed.tolist()}"
        )
    if feature_names is not None:
        if len(names) != numeric.shape[1] or len(names) != len(set(names)):
            raise ValueError("feature_names must be unique and match X_train columns")
        if hasattr(X, "columns") and list(map(str, X.columns)) != names:
            raise ValueError("X_train column order differs from feature_names")
    return X, labels


def fit_model_pipelines(
    X_train: Any,
    y_train: Sequence[int],
    *,
    feature_names: Sequence[str] | None = None,
    seed: int = RANDOM_SEED,
    pipelines: Mapping[str, Pipeline] | None = None,
) -> dict[str, Pipeline]:
    """Fit each fixed pipeline once on training windows only."""

    fitted, _ = fit_model_pipelines_timed(
        X_train,
        y_train,
        feature_names=feature_names,
        seed=seed,
        pipelines=pipelines,
    )
    return fitted


def fit_model_pipelines_timed(
    X_train: Any,
    y_train: Sequence[int],
    *,
    feature_names: Sequence[str] | None = None,
    seed: int = RANDOM_SEED,
    pipelines: Mapping[str, Pipeline] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, float]]:
    """Fit each pipeline once and return wall-clock training seconds per model."""

    features, labels = _validate_training_arrays(X_train, y_train, feature_names)
    selected = dict(pipelines) if pipelines is not None else build_model_pipelines(seed)
    missing = set(MODEL_SLUGS) - set(selected)
    extra = set(selected) - set(MODEL_SLUGS)
    if missing or extra:
        raise ValueError(
            f"Pipelines must match the eight model slugs; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    fitted: dict[str, Pipeline] = {}
    timings: dict[str, float] = {}
    for slug in MODEL_SLUGS:
        LOGGER.info("Fitting %s on %d training windows", slug, len(labels))
        model = selected[slug]
        started = perf_counter()
        model.fit(features, labels)
        timings[slug] = float(perf_counter() - started)
        classes = np.asarray(model.classes_, dtype=int)
        if not np.array_equal(classes, INTERNAL_CLASS_LABELS):
            raise AssertionError(f"{slug} fitted class order is {classes.tolist()}, not 0..3")
        fitted[slug] = model
        LOGGER.info("Fitted %s in %.3f seconds", slug, timings[slug])
    return fitted, timings


def save_model_pipelines(
    fitted_models: Mapping[str, Pipeline],
    saved_models_dir: str | Path,
) -> dict[str, Path]:
    """Save exactly one complete fitted pipeline under each canonical filename."""

    missing = set(MODEL_SLUGS) - set(fitted_models)
    extra = set(fitted_models) - set(MODEL_SLUGS)
    if missing or extra:
        raise ValueError(
            f"Fitted models must match MODEL_SLUGS; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    destination = Path(saved_models_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for slug in MODEL_SLUGS:
        model = fitted_models[slug]
        check_is_fitted(model)
        path = destination / MODEL_FILENAMES[slug]
        joblib.dump(model, path)
        paths[slug] = path
    return paths


def fit_and_save_model_pipelines(
    X_train: Any,
    y_train: Sequence[int],
    saved_models_dir: str | Path,
    *,
    feature_names: Sequence[str] | None = None,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, Pipeline], dict[str, Path]]:
    """Fit the eight fixed pipelines and persist them as joblib artifacts."""

    fitted = fit_model_pipelines(
        X_train,
        y_train,
        feature_names=feature_names,
        seed=seed,
    )
    return fitted, save_model_pipelines(fitted, saved_models_dir)


def aligned_predict_proba(model: Pipeline, X: Any) -> np.ndarray:
    """Return probabilities in canonical internal order 0, 1, 2, 3."""

    check_is_fitted(model)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Every benchmark pipeline must implement predict_proba")
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if raw.ndim != 2 or raw.shape[0] != len(X) or raw.shape[1] != len(classes):
        raise ValueError("Estimator returned an invalid probability matrix")
    aligned = np.zeros((len(X), len(INTERNAL_CLASS_LABELS)), dtype=float)
    for source_column, label in enumerate(classes):
        matches = np.flatnonzero(INTERNAL_CLASS_LABELS == label)
        if len(matches) != 1:
            raise ValueError(f"Estimator returned unknown internal class {label!r}")
        aligned[:, int(matches[0])] = raw[:, source_column]
    if not np.isfinite(aligned).all() or (aligned < -1e-12).any():
        raise ValueError("Estimator returned invalid probability values")
    aligned = np.clip(aligned, 0.0, 1.0)
    row_sums = aligned.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("At least one probability row has zero total mass")
    aligned /= row_sums
    return aligned


def model_parameter_manifest(models: Mapping[str, Pipeline]) -> dict[str, dict[str, Any]]:
    """Return exact classifier parameters for the reproducibility manifest."""

    manifest: dict[str, dict[str, Any]] = {}
    for slug in MODEL_SLUGS:
        if slug not in models:
            raise ValueError(f"Missing model {slug!r}")
        classifier = models[slug].named_steps["classifier"]
        manifest[slug] = classifier.get_params(deep=False)
    return manifest


# Backward-compatible script aliases; each delegates to the single implementation.
train_models = fit_model_pipelines
save_models = save_model_pipelines
fit_and_save_models = fit_and_save_model_pipelines


__all__ = [
    "INTERNAL_CLASS_LABELS",
    "MODEL_FILENAMES",
    "aligned_predict_proba",
    "build_model_pipelines",
    "build_models",
    "create_model_pipelines",
    "fit_and_save_model_pipelines",
    "fit_and_save_models",
    "fit_model_pipelines",
    "fit_model_pipelines_timed",
    "model_parameter_manifest",
    "save_model_pipelines",
    "save_models",
    "train_models",
]
