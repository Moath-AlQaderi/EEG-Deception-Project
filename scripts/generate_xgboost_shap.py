"""Generate out-of-fold SHAP explanations from the saved XGBoost fold pipelines.

This is a post-hoc explainability runner. It never fits an imputer, scaler, model,
cross-validation splitter, or EEG feature extractor. The stored fold assignments
and stored XGBoost out-of-fold predictions are treated as authoritative.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Any, Iterable, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("Honest", "Bluffer", "Liar", "Deceiver")
CLASS_COLUMNS = (
    "honest_mean_abs_shap",
    "bluffer_mean_abs_shap",
    "liar_mean_abs_shap",
    "deceiver_mean_abs_shap",
)
PROBABILITY_COLUMNS = (
    "probability_honest",
    "probability_bluffer",
    "probability_liar",
    "probability_deceiver",
)
EXPECTED_FOLDS = tuple(range(1, 11))
EXPECTED_CLASSES = np.arange(4, dtype=int)
FEATURE_PREFIX = "f__"
SHAP_CELL_TAG = "xgboost-oof-shap"
LOGGER = logging.getLogger("generate_xgboost_shap")

TIME_DOMAIN_FEATURES = {
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
}

FEATURE_LABEL_OVERRIDES = {
    "root_mean_square": "Root mean square",
    "peak_to_peak": "Peak-to-peak amplitude",
    "interquartile_range": "Interquartile range",
    "mean_absolute_value": "Mean absolute value",
    "waveform_length": "Waveform length",
    "zero_crossing_rate": "Zero-crossing rate",
    "hjorth_activity": "Hjorth activity",
    "hjorth_mobility": "Hjorth mobility",
    "hjorth_complexity": "Hjorth complexity",
    "delta_power": "Delta power",
    "theta_power": "Theta power",
    "alpha_power": "Alpha power",
    "beta_power": "Beta power",
    "total_power_1_30_hz": "Total power (1–30 Hz)",
    "relative_delta_power": "Relative delta power",
    "relative_theta_power": "Relative theta power",
    "relative_alpha_power": "Relative alpha power",
    "relative_beta_power": "Relative beta power",
    "theta_to_alpha_ratio": "Theta-to-alpha ratio",
    "theta_to_beta_ratio": "Theta-to-beta ratio",
    "spectral_entropy": "Spectral entropy",
    "dominant_frequency": "Dominant frequency",
    "median_frequency": "Median frequency",
    "standard_deviation": "Standard deviation",
}


class ValidationError(RuntimeError):
    """Raised when a provenance or explainability invariant is violated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/window_features.csv"),
        help="Existing window-level feature CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/minmax_10fold_trial_grouped"),
        help="Existing non-subject-wise MinMax 10-fold result directory.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("saved_models/minmax_10fold_trial_grouped/xgboost"),
        help="Directory containing fold_01.joblib through fold_10.joblib.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("EEG Deception Results.ipynb"),
        help="Existing results notebook to update after all outputs validate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Rows per TreeExplainer call; all eligible windows are still explained.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def absolute_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""

    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    """Return a project-relative display path whenever possible."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_columns(frame: pd.DataFrame, names: Iterable[str], source: Path) -> None:
    missing = sorted(set(names) - set(frame.columns))
    require(not missing, f"{display_path(source)} is missing columns: {missing}")


def normalize_multiclass_shap(
    raw_values: Any,
    *,
    n_rows: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Normalize documented multiclass SHAP layouts to rows × features × classes."""

    values = getattr(raw_values, "values", raw_values)
    if isinstance(values, (list, tuple)):
        require(
            len(values) == n_classes,
            f"Expected {n_classes} legacy SHAP class arrays, observed {len(values)}",
        )
        blocks = [np.asarray(block, dtype=np.float64) for block in values]
        unexpected = [block.shape for block in blocks if block.shape != (n_rows, n_features)]
        require(
            not unexpected,
            f"Legacy SHAP arrays have invalid shapes: {unexpected}",
        )
        normalized = np.stack(blocks, axis=-1)
    else:
        array = np.asarray(values, dtype=np.float64)
        target = (n_rows, n_features, n_classes)
        layouts = {
            target: (0, 1, 2),
            (n_classes, n_rows, n_features): (1, 2, 0),
            (n_rows, n_classes, n_features): (0, 2, 1),
        }
        require(
            array.shape in layouts,
            f"Unsupported multiclass SHAP shape {array.shape}; expected a layout of {target}",
        )
        normalized = np.transpose(array, layouts[array.shape])

    require(
        normalized.shape == (n_rows, n_features, n_classes),
        f"Canonical SHAP shape failure: {normalized.shape}",
    )
    require(np.isfinite(normalized).all(), "SHAP values contain NaN or infinite values")
    return normalized


def normalize_expected_value(raw_value: Any, n_classes: int) -> np.ndarray:
    values = np.asarray(raw_value, dtype=np.float64)
    if values.shape == (n_classes,):
        normalized = values
    elif values.shape == (1, n_classes):
        normalized = values[0]
    else:
        raise ValidationError(
            f"Unexpected multiclass TreeExplainer expected-value shape: {values.shape}"
        )
    require(np.isfinite(normalized).all(), "TreeExplainer expected values are non-finite")
    return normalized


def parse_feature_name(feature: str) -> tuple[str, str]:
    match = re.fullmatch(r"f__([^_]+)__(.+)", feature)
    require(match is not None, f"Feature name does not follow the trained schema: {feature}")
    return match.group(1), match.group(2)


def display_feature_name(feature: str) -> str:
    """Create a readable label without changing channel or feature meaning."""

    channel, measurement = parse_feature_name(feature)
    label = FEATURE_LABEL_OVERRIDES.get(
        measurement,
        measurement.replace("_", " ").capitalize(),
    )
    return f"{channel} – {label}"


def channel_region(channel: str) -> str:
    if channel.startswith("Fp"):
        return "frontopolar"
    if channel.startswith("F"):
        return "frontal"
    if channel.startswith("C"):
        return "central"
    if channel.startswith("T"):
        return "temporal"
    if channel.startswith("P"):
        return "parietal"
    if channel.startswith("O"):
        return "occipital"
    return "other"


def concentration_sentence(items: Sequence[str], noun: str) -> str:
    counts = Counter(items)
    leading, count = counts.most_common(1)[0]
    if count == len(items):
        return f"All three leading features are concentrated in the {leading} {noun}."
    if count >= 2:
        return f"Two of the three leading features are concentrated in the {leading} {noun}."
    return f"The three leading features are not concentrated in one {noun}; they span {', '.join(items)}."


def build_interpretation(top_three: pd.DataFrame) -> str:
    lines = [
        "### Concise interpretation",
        "",
        "The three most globally important features were:",
        "",
    ]
    channels: list[str] = []
    regions: list[str] = []
    families: list[str] = []
    for rank, row in enumerate(top_three.itertuples(index=False), start=1):
        channel, measurement = parse_feature_name(row.feature)
        channels.append(channel)
        regions.append(channel_region(channel))
        families.append(
            "time-domain" if measurement in TIME_DOMAIN_FEATURES else "frequency-domain"
        )
        class_values = np.asarray(
            [
                row.honest_mean_abs_shap,
                row.bluffer_mean_abs_shap,
                row.liar_mean_abs_shap,
                row.deceiver_mean_abs_shap,
            ],
            dtype=float,
        )
        class_index = int(np.argmax(class_values))
        lines.append(
            f"{rank}. **{display_feature_name(row.feature)}** "
            f"(`{row.feature}`): global mean |SHAP value| = "
            f"{row.global_mean_abs_shap:.6f}; the largest class-specific magnitude "
            f"was for **{CLASS_NAMES[class_index]}** ({class_values[class_index]:.6f})."
        )
    lines.extend(
        [
            "",
            concentration_sentence(channels, "channel"),
            concentration_sentence(regions, "electrode-region label"),
            concentration_sentence(families, "feature family"),
            "",
            "These magnitudes describe which inputs the fitted classifier relied on. "
            "They do not show direction of effect, physiological causality, or proof "
            "that a brain region generates deception.",
        ]
    )
    return "\n".join(lines)


def save_global_plot(top20: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    labels = [display_feature_name(name) for name in top20["feature"]]
    values = top20["global_mean_abs_shap"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.2, 7.1), facecolor="white")
    axis.set_facecolor("white")
    axis.barh(np.arange(len(top20)), values, color="#2F6C99", edgecolor="none")
    axis.set_yticks(np.arange(len(top20)), labels=labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Global mean |SHAP value|", fontsize=10)
    axis.set_ylabel("EEG feature", fontsize=10)
    axis.set_title("Global XGBoost Feature Importance", fontsize=11, pad=10)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.grid(axis="y", visible=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def cell_text_color(value: float, minimum: float, maximum: float) -> str:
    if maximum <= minimum:
        return "black"
    normalized = (value - minimum) / (maximum - minimum)
    return "white" if normalized >= 0.58 else "black"


def format_cell_value(value: float) -> str:
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def save_classwise_heatmap(
    top20: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    labels = [display_feature_name(name) for name in top20["feature"]]
    matrix = top20.loc[:, CLASS_COLUMNS].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.2, 8.2), facecolor="white")
    image = axis.pcolormesh(
        np.arange(matrix.shape[1] + 1),
        np.arange(matrix.shape[0] + 1),
        matrix,
        cmap="viridis",
        shading="flat",
        edgecolors="none",
        rasterized=False,
    )
    axis.set_xlim(0, matrix.shape[1])
    axis.set_ylim(matrix.shape[0], 0)
    axis.set_xticks(np.arange(4) + 0.5, labels=CLASS_NAMES, fontsize=9)
    axis.set_yticks(np.arange(len(top20)) + 0.5, labels=labels, fontsize=8)
    axis.set_xlabel("Behavioral role", fontsize=10)
    axis.set_ylabel("EEG feature", fontsize=10)
    axis.set_title("Class-Wise XGBoost SHAP Importance", fontsize=11, pad=10)
    minimum = float(matrix.min())
    maximum = float(matrix.max())
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            axis.text(
                column + 0.5,
                row + 0.5,
                format_cell_value(value),
                ha="center",
                va="center",
                fontsize=6.4,
                color=cell_text_color(value, minimum, maximum),
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Mean |SHAP value|", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def markdown_cell(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": [SHAP_CELL_TAG]},
        "source": source.splitlines(keepends=True),
    }


def code_cell(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": [SHAP_CELL_TAG]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def shap_notebook_cells(interpretation: str) -> list[dict[str, Any]]:
    introduction = """# XGBoost Explainability Using SHAP

SHAP values were calculated only for the **out-of-fold test windows** from the stored non-subject-wise MinMax 10-fold experiment. Each window was explained by the saved XGBoost model from the fold in which that window's trial was held out. The fitted fold-specific median imputer and MinMax scaler were reused without refitting.

Mean absolute SHAP values measure contribution magnitude. The global bar plot averages magnitudes across all explained windows and all four classes; the heatmap retains separate Honest, Bluffer, Liar, and Deceiver magnitudes. Because participants can occur in both training and test portions, this analysis is trial-grouped but **not subject-independent**.

> SHAP importance indicates model reliance. It is not causality, does not prove that a brain region generates deception, and does not show whether a feature increases or decreases a class output.
"""
    table_code = """SHAP_DIR = RESULT_DIR / "shap"
SHAP_TOP20_PATH = SHAP_DIR / "xgboost_top20_shap.csv"
SHAP_MAPPING_PATH = SHAP_DIR / "feature_name_mapping.csv"
required_shap_files = [
    SHAP_TOP20_PATH,
    SHAP_MAPPING_PATH,
    SHAP_DIR / "xgboost_global_top20_shap.png",
    SHAP_DIR / "xgboost_classwise_top20_shap_heatmap.png",
]
missing_shap_files = [path.as_posix() for path in required_shap_files if not path.is_file()]
if missing_shap_files:
    raise FileNotFoundError(f"Missing SHAP presentation files: {missing_shap_files}")

shap_top20 = pd.read_csv(SHAP_TOP20_PATH)
shap_mapping = pd.read_csv(SHAP_MAPPING_PATH)
display_name_by_feature = shap_mapping.set_index("feature")["display_feature"].to_dict()
shap_table = shap_top20.copy()
shap_table["feature"] = shap_table["feature"].map(display_name_by_feature)
shap_table = shap_table.rename(
    columns={
        "rank": "Rank",
        "feature": "EEG Feature",
        "global_mean_abs_shap": "Global mean |SHAP|",
        "honest_mean_abs_shap": "Honest mean |SHAP|",
        "bluffer_mean_abs_shap": "Bluffer mean |SHAP|",
        "liar_mean_abs_shap": "Liar mean |SHAP|",
        "deceiver_mean_abs_shap": "Deceiver mean |SHAP|",
    }
)
show_table(shap_table)
"""
    global_figure_code = """_ = show_existing_figure(
    SHAP_DIR / "xgboost_global_top20_shap.png",
    width=900,
)
"""
    heatmap_code = """_ = show_existing_figure(
    SHAP_DIR / "xgboost_classwise_top20_shap_heatmap.png",
    width=900,
)
"""
    return [
        markdown_cell("shap0001", introduction),
        markdown_cell(
            "shap0002",
            "## Top-20 out-of-fold SHAP importance table\n\n"
            "The same globally selected features and ordering are used in both figures.",
        ),
        code_cell("shap0003", table_code),
        markdown_cell("shap0004", "## Global top-20 SHAP importance"),
        code_cell("shap0005", global_figure_code),
        markdown_cell(
            "shap0006",
            "*Global mean absolute SHAP importance of the 20 most influential EEG "
            "features for the XGBoost classifier. Importance values were aggregated "
            "across the four behavioral classes using out-of-fold test-window explanations.*",
        ),
        markdown_cell("shap0007", "## Class-wise SHAP heatmap"),
        code_cell("shap0008", heatmap_code),
        markdown_cell(
            "shap0009",
            "*Class-wise mean absolute SHAP importance for the 20 globally most "
            "influential EEG features. Each column represents one behavioral role, "
            "and larger values indicate a stronger contribution of the feature to "
            "the corresponding XGBoost class output. SHAP values were computed using "
            "out-of-fold test windows.*",
        ),
        markdown_cell("shap0010", interpretation),
    ]


def update_notebook(notebook_path: Path, interpretation: str) -> Path:
    """Idempotently insert display-only SHAP cells before the existing final summary."""

    require(notebook_path.is_file(), f"Notebook not found: {display_path(notebook_path)}")
    backup_path = notebook_path.with_name(f"{notebook_path.stem}.backup{notebook_path.suffix}")
    if backup_path.is_file():
        LOGGER.info("Preserving existing notebook backup: %s", display_path(backup_path))
    else:
        shutil.copy2(notebook_path, backup_path)
        LOGGER.info("Created notebook backup: %s", display_path(backup_path))

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    require(
        notebook.get("nbformat") == 4 and isinstance(notebook.get("cells"), list),
        "The target notebook is not a supported nbformat-4 document",
    )
    retained_cells = [
        cell
        for cell in notebook["cells"]
        if SHAP_CELL_TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    insert_at = len(retained_cells)
    for index, cell in enumerate(retained_cells):
        source = "".join(cell.get("source", []))
        if source.startswith("## Notebook Execution Summary"):
            insert_at = index
            break
    cells = shap_notebook_cells(interpretation)
    notebook["cells"] = retained_cells[:insert_at] + cells + retained_cells[insert_at:]

    temporary_path = notebook_path.with_name(f".{notebook_path.name}.shap-update.tmp")
    temporary_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    json.loads(temporary_path.read_text(encoding="utf-8"))
    temporary_path.replace(notebook_path)
    return backup_path


def validate_parent_manifest(
    manifest: dict[str, Any],
    feature_path: Path,
) -> tuple[list[str], str]:
    require(manifest.get("status") == "complete", "Parent experiment manifest is not complete")
    verification = manifest.get("artifact_verification", {})
    require(
        verification.get("passed") is True,
        "Parent experiment artifact verification did not pass",
    )
    expected_hash = str(manifest.get("feature_source_sha256", ""))
    observed_hash = file_sha256(feature_path)
    require(
        observed_hash == expected_hash,
        "Feature CSV hash differs from the completed experiment manifest",
    )
    feature_summary = manifest.get("feature_summary", {})
    feature_names = [str(name) for name in feature_summary.get("feature_columns", [])]
    require(len(feature_names) == 496, "Parent manifest does not contain 496 feature names")
    return feature_names, observed_hash


def validate_assignments(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    assignment_path: Path,
) -> None:
    required = {
        "row_position",
        "window_id",
        "trial_id",
        "session_id",
        "participant_id",
        "role_id",
        "true_label",
        "fold",
    }
    require_columns(assignments, required, assignment_path)
    require(len(assignments) == len(features), "Fold assignments do not cover all feature rows")
    positions = assignments["row_position"].to_numpy(dtype=int)
    require(
        np.array_equal(np.sort(positions), np.arange(len(features))),
        "Fold row positions are not a unique complete 0-based sequence",
    )
    ordered = assignments.sort_values("row_position")
    require(
        np.array_equal(ordered["window_id"].astype(str), features["window_id"].astype(str)),
        "Fold assignment window IDs do not match feature CSV row positions",
    )
    require(
        np.array_equal(ordered["trial_id"].astype(str), features["trial_id"].astype(str)),
        "Fold assignment trial IDs do not match feature CSV row positions",
    )
    require(
        np.array_equal(ordered["role_id"].to_numpy(int), features["role_id"].to_numpy(int)),
        "Fold assignment role IDs do not match the feature CSV",
    )
    require(
        np.array_equal(ordered["true_label"].to_numpy(int), features["role_id"].to_numpy(int) - 1),
        "Stored internal labels do not match source role IDs",
    )
    require(
        set(assignments["fold"].astype(int)) == set(EXPECTED_FOLDS),
        "Fold assignments do not contain exactly folds 1 through 10",
    )
    require(
        assignments.groupby("fold").size().reindex(EXPECTED_FOLDS).eq(720).all(),
        "Each fold must contain exactly 720 held-out windows",
    )
    require(
        assignments.groupby("trial_id")["fold"].nunique().eq(1).all(),
        "A trial occurs in more than one held-out fold",
    )


def validate_oof_rows(
    assignments: pd.DataFrame,
    oof_rows: pd.DataFrame,
    oof_path: Path,
) -> pd.DataFrame:
    required = {
        "model",
        "fold",
        "source_row_index",
        "window_id",
        "trial_id",
        "true_label",
        "predicted_label",
        *PROBABILITY_COLUMNS,
    }
    require_columns(oof_rows, required, oof_path)
    xgboost_rows = oof_rows.loc[oof_rows["model"].eq("xgboost")].copy()
    require(len(xgboost_rows) == len(assignments), "XGBoost OOF rows do not cover all windows")
    require(
        xgboost_rows["source_row_index"].is_unique,
        "An XGBoost source row has more than one OOF prediction",
    )
    require(
        np.array_equal(
            np.sort(xgboost_rows["source_row_index"].to_numpy(int)),
            np.arange(len(assignments)),
        ),
        "XGBoost OOF source rows are not exactly 0 through N-1",
    )
    expected = assignments.set_index("row_position").sort_index()
    observed = xgboost_rows.set_index("source_row_index").sort_index()
    for column in ("fold", "window_id", "trial_id", "true_label"):
        require(
            np.array_equal(
                observed[column].astype(str).to_numpy(),
                expected[column].astype(str).to_numpy(),
            ),
            f"XGBoost OOF {column} differs from the authoritative fold assignment",
        )
    probabilities = observed.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
    require(
        np.isfinite(probabilities).all()
        and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
        "Stored XGBoost OOF probabilities are invalid",
    )
    return xgboost_rows


def validate_feature_schema(
    feature_names: Sequence[str],
    features: pd.DataFrame,
    feature_path: Path,
) -> None:
    require(len(feature_names) == len(set(feature_names)), "Feature names are not unique")
    require(
        all(name.startswith(FEATURE_PREFIX) for name in feature_names),
        "A non-EEG metadata column appears in the trained feature list",
    )
    require(
        all(name in features.columns for name in feature_names),
        "The feature CSV is missing a trained model input column",
    )
    csv_order = [str(name) for name in features.columns if str(name).startswith(FEATURE_PREFIX)]
    require(
        csv_order == list(feature_names),
        f"EEG feature order in {display_path(feature_path)} differs from the trained order",
    )
    schema_path = feature_path.parent / "feature_schema.json"
    if schema_path.is_file():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        require(
            [str(name) for name in schema.get("ordered_feature_names", [])]
            == list(feature_names),
            "feature_schema.json order differs from the trained pipeline order",
        )


def verify_model_manifest_hash(
    path: Path,
    parent_manifest: dict[str, Any],
) -> str:
    observed = file_sha256(path)
    key = display_path(path)
    expected = parent_manifest.get("artifact_sha256", {}).get(key)
    require(expected is not None, f"Parent manifest has no hash for {key}")
    require(observed == expected, f"Saved model hash differs from parent manifest: {key}")
    return observed


def create_importance_tables(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute_values = np.abs(shap_values)
    class_by_feature = absolute_values.mean(axis=0, dtype=np.float64).T
    require(
        class_by_feature.shape == (4, len(feature_names)),
        f"Class-wise importance has invalid shape: {class_by_feature.shape}",
    )
    global_importance = class_by_feature.mean(axis=0, dtype=np.float64)
    table = pd.DataFrame(
        {
            "feature": list(feature_names),
            "global_mean_abs_shap": global_importance,
            "honest_mean_abs_shap": class_by_feature[0],
            "bluffer_mean_abs_shap": class_by_feature[1],
            "liar_mean_abs_shap": class_by_feature[2],
            "deceiver_mean_abs_shap": class_by_feature[3],
        }
    )
    require(
        np.isfinite(table.select_dtypes(include="number").to_numpy()).all(),
        "Aggregated SHAP importance contains non-finite values",
    )
    require(
        (table.select_dtypes(include="number").to_numpy() >= 0).all(),
        "Aggregated mean absolute SHAP importance contains negative values",
    )
    table = table.sort_values(
        ["global_mean_abs_shap", "feature"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
    top20 = table.head(20).copy()
    require(len(top20) == 20 and top20["feature"].is_unique, "Top-20 features are invalid")
    return table, top20


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    require(args.seed == 42, "The completed experiment and SHAP pass require seed 42")
    require(args.batch_size > 0, "batch-size must be positive")
    require(
        sys.version_info >= (3, 12),
        "XGBoost 3.2 SHAP explanations require Python >=3.12 with SHAP 0.52+",
    )
    require(shap.__version__ == "0.52.0", "This validated runner requires SHAP 0.52.0")
    require(
        xgboost.__version__ == "3.2.0",
        "The saved fold classifiers require XGBoost 3.2.0",
    )
    require(
        sklearn.__version__ == "1.7.2",
        "The saved fold pipelines require scikit-learn 1.7.2",
    )

    feature_path = absolute_path(args.features)
    results_dir = absolute_path(args.results_dir)
    models_dir = absolute_path(args.models_dir)
    notebook_path = absolute_path(args.notebook)
    output_dir = results_dir / "shap"
    manifest_path = results_dir / "reproducibility_manifest.json"
    assignment_path = results_dir / "fold_assignments.csv"
    oof_path = results_dir / "oof_window_predictions.csv"
    model_path_table = results_dir / "fold_model_paths.csv"

    for path in (
        feature_path,
        manifest_path,
        assignment_path,
        oof_path,
        model_path_table,
        notebook_path,
    ):
        require(path.is_file(), f"Required input is missing: {display_path(path)}")
    require(models_dir.is_dir(), f"Model directory is missing: {display_path(models_dir)}")

    parent_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names, feature_hash = validate_parent_manifest(parent_manifest, feature_path)
    LOGGER.info("Loading feature table: %s", display_path(feature_path))
    features = pd.read_csv(feature_path)
    require(
        len(features) == 7200,
        f"Expected 7,200 window rows, observed {len(features):,}",
    )
    require_columns(features, {"window_id", "trial_id", "role_id"}, feature_path)
    validate_feature_schema(feature_names, features, feature_path)
    X = features.loc[:, feature_names]
    require(
        X.shape == (7200, 496),
        f"Expected model matrix shape (7200, 496), observed {X.shape}",
    )
    feature_array = X.to_numpy(dtype=float)
    require(
        not np.isinf(feature_array).any(),
        "Raw feature matrix contains infinite values",
    )

    assignments = pd.read_csv(assignment_path)
    validate_assignments(features, assignments, assignment_path)
    oof_columns = [
        "model",
        "fold",
        "source_row_index",
        "window_id",
        "trial_id",
        "true_label",
        "predicted_label",
        *PROBABILITY_COLUMNS,
    ]
    oof_rows = pd.read_csv(oof_path, usecols=oof_columns)
    xgboost_oof = validate_oof_rows(assignments, oof_rows, oof_path)

    model_records = pd.read_csv(model_path_table)
    require_columns(model_records, {"model", "fold", "model_path"}, model_path_table)
    xgboost_records = model_records.loc[model_records["model"].eq("xgboost")].copy()
    require(
        len(xgboost_records) == 10
        and set(xgboost_records["fold"].astype(int)) == set(EXPECTED_FOLDS),
        "fold_model_paths.csv does not contain exactly ten XGBoost fold records",
    )

    n_rows = len(features)
    n_features = len(feature_names)
    n_classes = len(CLASS_NAMES)
    shap_oof = np.full((n_rows, n_features, n_classes), np.nan, dtype=np.float32)
    explanation_count = np.zeros(n_rows, dtype=np.uint8)
    explained_fold = np.zeros(n_rows, dtype=np.uint8)
    model_hashes_before: dict[str, str] = {}
    additivity_rows: list[dict[str, float | int]] = []
    validation_checks: dict[str, bool] = {
        "parent_experiment_complete_and_verified": True,
        "feature_source_hash_matches_parent_manifest": True,
        "exact_496_feature_order_matches_all_sources": True,
        "stored_assignments_cover_each_window_exactly_once": True,
        "each_trial_has_exactly_one_test_fold": True,
        "xgboost_oof_rows_match_stored_assignments": True,
    }

    all_positions = np.arange(n_rows, dtype=int)
    for fold in EXPECTED_FOLDS:
        model_path = models_dir / f"fold_{fold:02d}.joblib"
        require(model_path.is_file(), f"Missing saved XGBoost fold model: {display_path(model_path)}")
        record = xgboost_records.loc[xgboost_records["fold"].astype(int).eq(fold)].iloc[0]
        recorded_path = Path(str(record["model_path"]))
        require(
            recorded_path.name == model_path.name and recorded_path.parent.name == "xgboost",
            f"Fold-model record does not correspond to XGBoost fold {fold}",
        )
        model_hashes_before[display_path(model_path)] = verify_model_manifest_hash(
            model_path,
            parent_manifest,
        )

        test_positions = np.sort(
            assignments.loc[
                assignments["fold"].astype(int).eq(fold),
                "row_position",
            ].to_numpy(dtype=int)
        )
        train_positions = np.setdiff1d(all_positions, test_positions, assume_unique=True)
        require(len(test_positions) == 720 and len(train_positions) == 6480, f"Fold {fold} row counts are invalid")
        test_trials = set(features.iloc[test_positions]["trial_id"].astype(str))
        train_trials = set(features.iloc[train_positions]["trial_id"].astype(str))
        require(
            not (test_trials & train_trials),
            f"Fold {fold} has trial overlap between training and test portions",
        )
        require(
            np.all(explanation_count[test_positions] == 0),
            f"Fold {fold} contains rows that were already explained",
        )

        LOGGER.info(
            "Fold %02d: loading saved pipeline and explaining %d OOF windows",
            fold,
            len(test_positions),
        )
        pipeline = joblib.load(model_path)
        require(
            list(pipeline.named_steps) == ["imputer", "scaler", "classifier"],
            f"Fold {fold} pipeline steps are not imputer → scaler → classifier",
        )
        require(
            list(map(str, pipeline.feature_names_in_)) == feature_names,
            f"Fold {fold} feature order differs from the trained order",
        )
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        classifier = pipeline.named_steps["classifier"]
        require(
            int(scaler.n_samples_seen_) == len(train_positions),
            f"Fold {fold} scaler was not fitted on exactly 6,480 training windows",
        )
        require(
            np.array_equal(np.asarray(pipeline.classes_, dtype=int), EXPECTED_CLASSES)
            and np.array_equal(np.asarray(classifier.classes_, dtype=int), EXPECTED_CLASSES),
            f"Fold {fold} classifier class order is not [0, 1, 2, 3]",
        )
        require(
            int(classifier.n_features_in_) == n_features,
            f"Fold {fold} classifier input width is not {n_features}",
        )

        X_test_raw = X.iloc[test_positions]
        X_test_imputed = imputer.transform(X_test_raw)
        X_test_model = scaler.transform(X_test_imputed)
        require(
            X_test_model.shape == (len(test_positions), n_features)
            and np.isfinite(X_test_model).all(),
            f"Fold {fold} transformed test matrix is invalid",
        )

        stored_fold = (
            xgboost_oof.loc[xgboost_oof["fold"].astype(int).eq(fold)]
            .set_index("source_row_index")
            .loc[test_positions]
        )
        observed_labels = np.asarray(classifier.predict(X_test_model), dtype=int)
        observed_probabilities = np.asarray(
            classifier.predict_proba(X_test_model),
            dtype=float,
        )
        require(
            np.array_equal(
                observed_labels,
                stored_fold["predicted_label"].to_numpy(dtype=int),
            ),
            f"Fold {fold} reloaded classifier predictions differ from stored OOF predictions",
        )
        require(
            np.allclose(
                observed_probabilities,
                stored_fold.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float),
                atol=1e-7,
                rtol=1e-6,
            ),
            f"Fold {fold} reloaded probabilities differ from stored OOF probabilities",
        )

        explainer = shap.TreeExplainer(
            classifier,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
        # XGBoost explainers may replace the loader's initial vector base score
        # with the ensemble expected value on the first explanation call.
        fold_expected_value: np.ndarray | None = None
        fold_additivity_errors: list[np.ndarray] = []
        fold_values = np.empty(
            (len(test_positions), n_features, n_classes),
            dtype=np.float32,
        )
        for start in range(0, len(test_positions), args.batch_size):
            stop = min(start + args.batch_size, len(test_positions))
            batch = X_test_model[start:stop]
            raw_values = explainer.shap_values(batch, check_additivity=False)
            normalized = normalize_multiclass_shap(
                raw_values,
                n_rows=len(batch),
                n_features=n_features,
                n_classes=n_classes,
            )
            expected_value = normalize_expected_value(
                explainer.expected_value,
                n_classes,
            )
            if fold_expected_value is None:
                fold_expected_value = expected_value.copy()
            else:
                require(
                    np.allclose(expected_value, fold_expected_value, atol=1e-12),
                    f"Fold {fold} TreeExplainer expected value changed across batches",
                )
            margins = np.asarray(
                classifier.predict(batch, output_margin=True),
                dtype=np.float64,
            )
            require(
                margins.shape == (len(batch), n_classes),
                f"Fold {fold} raw-margin prediction shape is invalid: {margins.shape}",
            )
            reconstructed = normalized.sum(axis=1, dtype=np.float64) + expected_value
            fold_additivity_errors.append(np.abs(reconstructed - margins))
            fold_values[start:stop] = normalized.astype(np.float32, copy=False)

        errors = np.concatenate(fold_additivity_errors, axis=0)
        maximum_error = float(errors.max())
        require(
            maximum_error <= 1e-4,
            f"Fold {fold} Tree SHAP additivity error is too large: {maximum_error:.6g}",
        )
        additivity_rows.append(
            {
                "fold": fold,
                "mean_abs_error": float(errors.mean()),
                "p99_abs_error": float(np.quantile(errors, 0.99)),
                "max_abs_error": maximum_error,
            }
        )
        shap_oof[test_positions] = fold_values
        explanation_count[test_positions] += 1
        explained_fold[test_positions] = fold
        del classifier, explainer, fold_values, pipeline, X_test_imputed, X_test_model

    require(
        np.all(explanation_count == 1),
        "Each eligible OOF window was not explained exactly once",
    )
    authoritative_fold_by_row = (
        assignments.sort_values("row_position")["fold"].to_numpy(dtype=np.uint8)
    )
    require(
        np.array_equal(explained_fold, authoritative_fold_by_row),
        "An explained row was not assigned to its authoritative test-fold model",
    )
    require(
        shap_oof.shape == (7200, 496, 4) and np.isfinite(shap_oof).all(),
        f"Final SHAP tensor is invalid: shape={shap_oof.shape}",
    )
    model_hashes_after = {
        path: file_sha256(PROJECT_ROOT / path)
        for path in model_hashes_before
    }
    require(
        model_hashes_after == model_hashes_before,
        "A saved XGBoost fold model changed during SHAP generation",
    )
    validation_checks.update(
        {
            "exactly_four_classes_in_consistent_order": True,
            "all_ten_saved_fold_pipelines_reused_without_refitting": True,
            "zero_trial_overlap_within_every_fold": True,
            "reloaded_predictions_match_stored_oof_predictions": True,
            "each_oof_window_explained_exactly_once": True,
            "final_shap_shape_is_7200_by_496_by_4": True,
            "tree_shap_raw_margin_additivity_max_error_le_1e-4": True,
            "saved_model_hashes_unchanged": True,
        }
    )

    all_features, top20 = create_importance_tables(shap_oof, feature_names)
    require(
        top20["feature"].tolist() == all_features.head(20)["feature"].tolist(),
        "The top-20 table order is not the global importance order",
    )
    validation_checks.update(
        {
            "all_importance_values_finite_and_nonnegative": True,
            "exactly_20_unique_features_selected": True,
            "same_top20_order_used_for_bar_plot_and_heatmap": True,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "xgboost_all_feature_shap.csv"
    top20_csv = output_dir / "xgboost_top20_shap.csv"
    mapping_csv = output_dir / "feature_name_mapping.csv"
    additivity_csv = output_dir / "xgboost_shap_additivity_checks.csv"
    global_png = output_dir / "xgboost_global_top20_shap.png"
    global_pdf = output_dir / "xgboost_global_top20_shap.pdf"
    heatmap_png = output_dir / "xgboost_classwise_top20_shap_heatmap.png"
    heatmap_pdf = output_dir / "xgboost_classwise_top20_shap_heatmap.pdf"
    summary_json = output_dir / "xgboost_shap_summary.json"

    all_features.to_csv(all_csv, index=False)
    top20.to_csv(top20_csv, index=False)
    pd.DataFrame(
        {
            "feature": feature_names,
            "display_feature": [display_feature_name(name) for name in feature_names],
        }
    ).to_csv(mapping_csv, index=False)
    pd.DataFrame(additivity_rows).to_csv(additivity_csv, index=False)
    save_global_plot(top20, global_png, global_pdf)
    save_classwise_heatmap(top20, heatmap_png, heatmap_pdf)

    for path in (all_csv, top20_csv, mapping_csv, global_png, global_pdf, heatmap_png, heatmap_pdf):
        require(path.is_file() and path.stat().st_size > 0, f"Output was not saved: {display_path(path)}")

    interpretation = build_interpretation(top20.head(3))
    backup_path = update_notebook(notebook_path, interpretation)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    shap_cells = [
        cell
        for cell in notebook["cells"]
        if SHAP_CELL_TAG in cell.get("metadata", {}).get("tags", [])
    ]
    require(len(shap_cells) == 10, "Notebook SHAP section was not inserted exactly once")

    top_three_records: list[dict[str, Any]] = []
    for row in top20.head(3).itertuples(index=False):
        class_values = np.asarray(
            [
                row.honest_mean_abs_shap,
                row.bluffer_mean_abs_shap,
                row.liar_mean_abs_shap,
                row.deceiver_mean_abs_shap,
            ]
        )
        largest_index = int(np.argmax(class_values))
        top_three_records.append(
            {
                "feature": row.feature,
                "display_feature": display_feature_name(row.feature),
                "global_mean_abs_shap": float(row.global_mean_abs_shap),
                "largest_class": CLASS_NAMES[largest_index],
                "largest_class_mean_abs_shap": float(class_values[largest_index]),
            }
        )

    summary = {
        "status": "complete",
        "method": {
            "explainer": "shap.TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw XGBoost class margins",
            "importance": "mean absolute SHAP value",
            "background_data": "none",
            "subset_used": False,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
        "source": {
            "features": display_path(feature_path),
            "features_sha256": feature_hash,
            "fold_assignments": display_path(assignment_path),
            "oof_window_predictions": display_path(oof_path),
            "models_directory": display_path(models_dir),
            "parent_manifest": display_path(manifest_path),
            "notebook": display_path(notebook_path),
            "notebook_backup_preserved": display_path(backup_path),
        },
        "coverage": {
            "out_of_fold_windows_explained": int(n_rows),
            "folds_processed": len(EXPECTED_FOLDS),
            "windows_per_fold": {
                str(fold): int((explained_fold == fold).sum()) for fold in EXPECTED_FOLDS
            },
            "participants_may_overlap_across_folds": True,
            "trial_overlap_within_fold": 0,
            "each_window_explained_exactly_once": True,
        },
        "shap_array_shape": list(map(int, shap_oof.shape)),
        "input_feature_count": int(n_features),
        "class_order": [
            {"internal_label": index, "class_name": name}
            for index, name in enumerate(CLASS_NAMES)
        ],
        "top20_features": top20["feature"].tolist(),
        "top_three": top_three_records,
        "additivity": additivity_rows,
        "validation_checks": validation_checks,
        "no_model_retrained": True,
        "only_out_of_fold_test_windows_explained": True,
        "model_sha256": model_hashes_before,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
        "outputs": {
            "top20_csv": display_path(top20_csv),
            "all_feature_csv": display_path(all_csv),
            "feature_name_mapping": display_path(mapping_csv),
            "additivity_checks": display_path(additivity_csv),
            "global_bar_png": display_path(global_png),
            "global_bar_pdf": display_path(global_pdf),
            "classwise_heatmap_png": display_path(heatmap_png),
            "classwise_heatmap_pdf": display_path(heatmap_pdf),
            "updated_notebook": display_path(notebook_path),
        },
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("XGBOOST OUT-OF-FOLD SHAP SUMMARY")
    print(f"Out-of-fold windows explained: {n_rows}")
    print(f"Folds processed: {len(EXPECTED_FOLDS)}")
    print(f"SHAP array shape: {tuple(shap_oof.shape)}")
    print(f"Input features: {n_features}")
    print("Top 20 feature names:")
    for row in top20.itertuples(index=False):
        print(f"  {row.rank:2d}. {row.feature}")
    print("Top three global features:")
    for item in top_three_records:
        print(
            f"  {item['feature']}: {item['global_mean_abs_shap']:.9f} "
            f"(largest class magnitude: {item['largest_class']}, "
            f"{item['largest_class_mean_abs_shap']:.9f})"
        )
    print(f"Global bar plot: {display_path(global_png)}")
    print(f"Class-wise heatmap: {display_path(heatmap_png)}")
    print(f"SHAP top-20 CSV: {display_path(top20_csv)}")
    print(f"Updated notebook: {display_path(notebook_path)}")
    print("No model was retrained: confirmed")
    print("Explanations used only out-of-fold test windows: confirmed")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        LOGGER.error("Validation failed: %s", error)
        raise SystemExit(2) from error
