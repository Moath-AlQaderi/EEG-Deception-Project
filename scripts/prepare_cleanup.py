"""Inventory and back up explicit cleanup targets for the final MinMax project.

This utility never deletes or moves pre-existing project content. Deletion is
deliberately performed later from the reviewed cleanup_manifest.csv with
explicit PowerShell paths.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import shutil
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = PROJECT_ROOT / "cleanup_manifest.csv"
MANIFEST_MD = PROJECT_ROOT / "cleanup_manifest.md"
BACKUP_ZIP = PROJECT_ROOT / "project_cleanup_backup.zip"
NOTEBOOK = PROJECT_ROOT / "EEG Deception Results.ipynb"
NOTEBOOK_BACKUP = PROJECT_ROOT / "EEG Deception Results.pre_cleanup_backup.ipynb"

FINAL_RESULTS_PREFIX = "outputs/minmax_10fold_trial_grouped"
FINAL_MODELS_PREFIX = "saved_models/minmax_10fold_trial_grouped"
DATASET_PREFIX = "Dataset_32302035"

COMMON_ROOT_OUTPUTS = {
    "outputs/window_features.csv",
    "outputs/feature_schema.json",
    "outputs/dataset_metadata.csv",
    "outputs/dataset_summary.json",
    "outputs/preprocessing_qc.csv",
}

UNCERTAIN_PATHS = {
    "scripts/run_ml_experiment.py": (
        "Legacy holdout runner, retained because it imports shared feature-generation "
        "modules and documents how the preserved feature table was originally produced."
    ),
    "src/evaluation.py": (
        "Legacy evaluation module, retained because a preserved contract test imports "
        "its trial-probability aggregation function."
    ),
    "tests/test_pipeline_contracts.py": (
        "Mixed legacy/shared contract tests; retained because most tests validate the "
        "preserved preprocessing, windowing, feature, and model components."
    ),
    f"{FINAL_RESULTS_PREFIX}/comparison_with_subjectwise.csv": (
        "Historical comparison is not displayed by the final notebook, but it is hashed "
        "by the preserved final MinMax reproducibility manifest."
    ),
}

PLANNED_KEEP_FILES = {
    "cleanup_manifest.csv",
    "cleanup_manifest.md",
    "cleanup_report.md",
    "project_cleanup_backup.zip",
    "EEG Deception Results.pre_cleanup_backup.ipynb",
}

PROTECTED_EXACT = {
    "EEG Deception Results.ipynb",
    "EEG Deception Results.backup.ipynb",
    "EEG Deception Results.pre_cleanup_backup.ipynb",
    "outputs/window_features.csv",
    "outputs/feature_schema.json",
    "outputs/dataset_metadata.csv",
    "outputs/dataset_summary.json",
    "outputs/preprocessing_qc.csv",
    "scripts/run_minmax_10fold.py",
    "scripts/generate_xgboost_shap.py",
    "scripts/prepare_cleanup.py",
    "src/minmax_cv.py",
    "src/models.py",
    "src/utils.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "cleanup_manifest.csv",
    "cleanup_manifest.md",
    "cleanup_report.md",
    "project_cleanup_backup.zip",
    "A Multi-Class EEG Dataset for Behavioral Roles in Deception Honesty, Bluffing, Lying, and Deceiving.pdf",
}

PROTECTED_PREFIXES = {
    ".git",
    ".venv",
    ".shap-venv",
    "cache",
    DATASET_PREFIX,
    FINAL_RESULTS_PREFIX,
    FINAL_MODELS_PREFIX,
}

REQUIRED_FINAL_RESULT_FILES = {
    "artifact_verification.json",
    "comparison_with_subjectwise.csv",
    "confusion_matrix_decision_tree.png",
    "confusion_matrix_knn.png",
    "confusion_matrix_logistic_regression.png",
    "confusion_matrix_mlp.png",
    "confusion_matrix_naive_bayes.png",
    "confusion_matrix_random_forest.png",
    "confusion_matrix_svm.png",
    "confusion_matrix_xgboost.png",
    "experiment_manifest.json",
    "experiment_summary.txt",
    "fold_assignments.csv",
    "fold_metrics.csv",
    "fold_model_paths.csv",
    "fold_participant_overlap.csv",
    "model_comparison.csv",
    "model_comparison.png",
    "oof_trial_predictions.csv",
    "oof_window_predictions.csv",
    "per_class_trial_metrics.csv",
    "pooled_trial_metrics.csv",
    "pooled_window_metrics.csv",
    "reproducibility_manifest.json",
    "run.log",
    "shap/feature_name_mapping.csv",
    "shap/xgboost_all_feature_shap.csv",
    "shap/xgboost_classwise_top20_shap_heatmap.pdf",
    "shap/xgboost_classwise_top20_shap_heatmap.png",
    "shap/xgboost_global_top20_shap.pdf",
    "shap/xgboost_global_top20_shap.png",
    "shap/xgboost_shap_additivity_checks.csv",
    "shap/xgboost_shap_summary.json",
    "shap/xgboost_top20_shap.csv",
}


@dataclass(frozen=True)
class Classification:
    label: str
    reason: str
    experiment: str
    action: str


def normalized_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def is_same_or_descendant(candidate: str, parent: str) -> bool:
    candidate_lower = candidate.replace("\\", "/").rstrip("/").lower()
    parent_lower = parent.replace("\\", "/").rstrip("/").lower()
    return candidate_lower == parent_lower or candidate_lower.startswith(
        parent_lower + "/"
    )


def validate_manifest_relative(relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if (
        not normalized
        or Path(normalized).is_absolute()
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
        or ":" in normalized
    ):
        raise RuntimeError(f"Unsafe manifest path: {relative!r}")
    candidate = (PROJECT_ROOT / Path(*normalized.split("/"))).resolve()
    root = PROJECT_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Manifest path escapes project root: {relative!r}") from error
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(relative: str, is_directory: bool) -> Classification:
    parts = relative.split("/")
    first = parts[0]

    if relative in PLANNED_KEEP_FILES:
        return Classification(
            "KEEP",
            "Cleanup provenance or rollback artifact required by the cleanup procedure.",
            "Project cleanup",
            "preserve",
        )
    if first in {".venv", ".shap-venv"}:
        return Classification(
            "KEEP",
            "Active project environment; excluded from cleanup and required for the "
            "classical or SHAP workflow.",
            "Final MinMax/SHAP environments",
            "preserve",
        )
    if first == ".git":
        return Classification(
            "KEEP",
            "Version-control metadata must be preserved.",
            "Project configuration",
            "preserve",
        )
    if first == DATASET_PREFIX:
        return Classification(
            "KEEP",
            "Original EEG dataset, session material, MATLAB recordings, or experimental "
            "documentation.",
            "Original research material",
            "preserve",
        )
    if relative == (
        "A Multi-Class EEG Dataset for Behavioral Roles in Deception Honesty, "
        "Bluffing, Lying, and Deceiving.pdf"
    ):
        return Classification(
            "KEEP",
            "Original research paper.",
            "Original research material",
            "preserve",
        )
    if first == "cache":
        return Classification(
            "KEEP",
            "Processed epochs/windows and indices used to produce the preserved final "
            "feature table.",
            "Shared final feature preparation",
            "preserve",
        )
    if relative == FINAL_RESULTS_PREFIX or relative.startswith(FINAL_RESULTS_PREFIX + "/"):
        if relative in UNCERTAIN_PATHS:
            return Classification(
                "UNCERTAIN",
                UNCERTAIN_PATHS[relative],
                "Historical provenance inside final MinMax results",
                "retain_no_delete",
            )
        return Classification(
            "KEEP",
            "Final non-subject-wise MinMax 10-fold results, OOF records, figures, "
            "manifests, or SHAP output.",
            "Final MinMax/SHAP experiment",
            "preserve",
        )
    if relative == FINAL_MODELS_PREFIX or relative.startswith(FINAL_MODELS_PREFIX + "/"):
        return Classification(
            "KEEP",
            "Final fold-specific pipeline required for OOF verification or SHAP.",
            "Final MinMax 10-fold experiment",
            "preserve",
        )
    if first == "partial_runs":
        return Classification(
            "DELETE",
            "Explicitly aborted MinMax preflight containing incomplete duplicate models "
            "and a truncated log.",
            "Aborted/temporary experiment",
            "archive_then_delete",
        )
    if first == ".pytest_cache" or "__pycache__" in parts:
        return Classification(
            "DELETE",
            "Generated test or Python bytecode cache; safely recreated.",
            "Generated cache",
            "archive_then_delete",
        )
    if first == "outputs":
        if len(parts) == 1:
            return Classification(
                "KEEP",
                "Mixed container holding final inputs and final result subdirectory.",
                "Project results container",
                "preserve",
            )
        if relative in COMMON_ROOT_OUTPUTS:
            return Classification(
                "KEEP",
                "Shared dataset/feature metadata or the exact feature input required by "
                "the final MinMax and SHAP scripts.",
                "Shared final feature preparation",
                "preserve",
            )
        if len(parts) == 2 and not is_directory:
            return Classification(
                "DELETE",
                "Root-level artifact from the obsolete subject-wise 80/20 holdout; the "
                "final notebook and SHAP workflow load only the MinMax subdirectory.",
                "Obsolete subject-wise 80/20 experiment",
                "archive_then_delete",
            )
    if first == "saved_models":
        if len(parts) == 1:
            return Classification(
                "KEEP",
                "Mixed container holding the final fold-specific model directory.",
                "Project model container",
                "preserve",
            )
        if len(parts) == 2 and relative.endswith(".joblib"):
            return Classification(
                "DELETE",
                "Single holdout model from the obsolete subject-wise pipeline; superseded "
                "by the 10 final fold-specific models for this classifier.",
                "Obsolete subject-wise 80/20 experiment",
                "archive_then_delete",
            )
    if relative in UNCERTAIN_PATHS:
        return Classification(
            "UNCERTAIN",
            UNCERTAIN_PATHS[relative],
            "Shared/legacy source",
            "retain_no_delete",
        )
    if first in {"scripts", "src", "tests"}:
        return Classification(
            "KEEP",
            "Source or test retained for final feature provenance, MinMax execution, "
            "SHAP generation, saved-model loading, or lightweight validation.",
            "Final/shared implementation",
            "preserve",
        )
    if relative in {
        "EEG Deception Results.ipynb",
        "EEG Deception Results.backup.ipynb",
        "README.md",
        "requirements.txt",
        ".gitignore",
    }:
        return Classification(
            "KEEP",
            "Final notebook, retained backup, documentation, dependency declaration, or "
            "project configuration.",
            "Final project",
            "preserve",
        )
    if is_directory and first in {"outputs", "saved_models"}:
        return Classification(
            "UNCERTAIN",
            "Unrecognized entry inside a mixed scientific artifact container.",
            "Unknown",
            "retain_no_delete",
        )
    return Classification(
        "UNCERTAIN",
        "Purpose is not confidently attributable to an obsolete experiment.",
        "Unknown",
        "retain_no_delete",
    )


def entry_type(path: Path, relative: str, is_directory: bool) -> str:
    if is_directory:
        if relative.startswith((".venv/", ".shap-venv/")):
            return "Environment directory"
        if "__pycache__" in relative or relative.startswith(".pytest_cache"):
            return "Cache directory"
        if relative.startswith(DATASET_PREFIX):
            return "Dataset directory"
        if relative.startswith(("outputs/", "saved_models/", "partial_runs/")):
            return "Artifact directory"
        return "Directory"
    suffix = path.suffix.lower()
    mapping = {
        ".py": "Python source",
        ".pyc": "Python bytecode cache",
        ".ipynb": "Jupyter notebook",
        ".csv": "CSV table",
        ".json": "JSON metadata",
        ".npy": "NumPy cache",
        ".npz": "NumPy cache",
        ".mat": "MATLAB EEG data",
        ".pdf": "PDF document/figure",
        ".png": "PNG figure",
        ".joblib": "Trained model",
        ".txt": "Text result",
        ".log": "Log",
        ".md": "Documentation",
    }
    if path.name == ".gitignore":
        return "Configuration"
    return mapping.get(suffix, "Environment/support file" if relative.startswith((".venv/", ".shap-venv/")) else "Other file")


def scan_entries() -> list[dict[str, object]]:
    paths = sorted(
        (
            path
            for path in PROJECT_ROOT.rglob("*")
            if normalized_relative(path) not in PLANNED_KEEP_FILES
            and not normalized_relative(path).startswith("_cleanup_archive/")
        ),
        key=lambda item: normalized_relative(item).lower(),
    )
    file_sizes: dict[Path, int] = {}
    directory_sizes: dict[Path, int] = {}
    for path in paths:
        if path.is_file():
            size = path.stat().st_size
            file_sizes[path] = size
            parent = path.parent
            while parent != PROJECT_ROOT:
                directory_sizes[parent] = directory_sizes.get(parent, 0) + size
                parent = parent.parent

    rows: list[dict[str, object]] = []
    for path in paths:
        relative = normalized_relative(path)
        is_directory = path.is_dir()
        decision = classify(relative, is_directory)
        rows.append(
            {
                "path": relative,
                "type": entry_type(path, relative, is_directory),
                "size": directory_sizes.get(path, 0) if is_directory else file_sizes.get(path, 0),
                "classification": decision.label,
                "reason": decision.reason,
                "related experiment": decision.experiment,
                "action": decision.action,
                "sha256": (
                    sha256(path)
                    if not is_directory and decision.label == "DELETE"
                    else ""
                ),
            }
        )
    for relative in sorted(PLANNED_KEEP_FILES):
        rows.append(
            {
                "path": relative,
                "type": "Planned cleanup artifact",
                "size": (PROJECT_ROOT / relative).stat().st_size if (PROJECT_ROOT / relative).is_file() else 0,
                "classification": "KEEP",
                "reason": "Cleanup provenance or rollback artifact required by the cleanup procedure.",
                "related experiment": "Project cleanup",
                "action": "preserve",
                "sha256": "",
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]).lower())


def deletion_roots(rows: list[dict[str, object]]) -> list[str]:
    deleted = {str(row["path"]) for row in rows if row["classification"] == "DELETE"}
    roots: list[str] = []
    for relative in sorted(deleted, key=lambda value: (value.count("/"), value.lower())):
        parts = relative.split("/")
        ancestors = {"/".join(parts[:index]) for index in range(1, len(parts))}
        if not (ancestors & deleted):
            roots.append(relative)
    return roots


def protected_guard(rows: list[dict[str, object]]) -> None:
    deleted = [str(row["path"]) for row in rows if row["classification"] == "DELETE"]
    protected_exact = set(PROTECTED_EXACT)
    for source_root in ("scripts", "src", "tests"):
        protected_exact.update(
            normalized_relative(path)
            for path in (PROJECT_ROOT / source_root).rglob("*.py")
            if path.is_file()
        )
    violations: list[str] = []
    for relative in deleted:
        candidate = validate_manifest_relative(relative)
        current = candidate
        while current != PROJECT_ROOT:
            if current.exists():
                attributes = getattr(current.lstat(), "st_file_attributes", 0)
                if current.is_symlink() or attributes & 0x400:
                    raise RuntimeError(
                        f"Deletion path traverses a link/reparse point: {relative}"
                    )
            current = current.parent
        conflicts = any(
            is_same_or_descendant(relative, protected)
            or is_same_or_descendant(protected, relative)
            for protected in protected_exact | PROTECTED_PREFIXES
        )
        if conflicts or relative.lower().endswith(".mat"):
            violations.append(relative)
    if violations:
        raise RuntimeError(f"Protected paths entered the deletion list: {violations}")

    required_files = {
        "EEG Deception Results.ipynb",
        "outputs/window_features.csv",
        "outputs/feature_schema.json",
        "outputs/dataset_metadata.csv",
        "outputs/dataset_summary.json",
        "outputs/preprocessing_qc.csv",
        "scripts/run_minmax_10fold.py",
        "scripts/generate_xgboost_shap.py",
        "src/minmax_cv.py",
        "src/models.py",
        "src/utils.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
        (
            "A Multi-Class EEG Dataset for Behavioral Roles in Deception Honesty, "
            "Bluffing, Lying, and Deceiving.pdf"
        ),
        f"{DATASET_PREFIX}/Questions.pdf",
        "cache/epochs_unnormalized.npy",
        "cache/feature_stage_manifest.json",
        "cache/preprocessing_manifest.json",
        "cache/trial_index.csv",
        "cache/window_index.csv",
        "cache/windowing_manifest.json",
        "cache/windows.npy",
        ".venv/Scripts/python.exe",
        ".shap-venv/Scripts/python.exe",
        *{
            f"{FINAL_RESULTS_PREFIX}/{relative}"
            for relative in REQUIRED_FINAL_RESULT_FILES
        },
    }
    bad_required = []
    for relative in sorted(required_files):
        path = validate_manifest_relative(relative)
        if not path.is_file() or path.stat().st_size <= 0:
            bad_required.append(relative)
    required_directories = {
        DATASET_PREFIX,
        "cache",
        FINAL_RESULTS_PREFIX,
        FINAL_MODELS_PREFIX,
        ".venv",
        ".shap-venv",
    }
    bad_required.extend(
        relative
        for relative in sorted(required_directories)
        if not validate_manifest_relative(relative).is_dir()
    )
    if bad_required:
        raise RuntimeError(
            f"Required final files/directories are missing, empty, or wrong type: "
            f"{bad_required}"
        )

    matlab_files = sorted((PROJECT_ROOT / DATASET_PREFIX).rglob("*.mat"))
    if len(matlab_files) != 952 or any(
        not path.is_file() or path.stat().st_size <= 0 for path in matlab_files
    ):
        raise RuntimeError(
            f"Expected 952 non-empty original MATLAB files; found {len(matlab_files)}"
        )
    model_files = sorted((PROJECT_ROOT / FINAL_MODELS_PREFIX).rglob("*.joblib"))
    xgboost_files = sorted(
        (PROJECT_ROOT / FINAL_MODELS_PREFIX / "xgboost").glob("fold_*.joblib")
    )
    if (
        len(model_files) != 80
        or len(xgboost_files) != 10
        or any(path.stat().st_size <= 0 for path in model_files)
    ):
        raise RuntimeError(
            "Final saved-model contract failed: expected 80 non-empty fold models "
            "including 10 XGBoost fold models"
        )


def write_manifest(rows: list[dict[str, object]]) -> None:
    fields = [
        "path",
        "type",
        "size",
        "classification",
        "reason",
        "related experiment",
        "action",
        "sha256",
    ]
    with MANIFEST_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    file_rows = [row for row in rows if "directory" not in str(row["type"]).lower() and row["type"] != "Planned cleanup artifact"]
    deleted_files = [row for row in file_rows if row["classification"] == "DELETE"]
    uncertain_files = [row for row in file_rows if row["classification"] == "UNCERTAIN"]
    roots = deletion_roots(rows)
    lines = [
        "# Project Cleanup Dry-Run Manifest",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "No files were moved or deleted while creating this manifest.",
        "",
        "## Counts",
        "",
        f"- Files to keep: {sum(row['classification'] == 'KEEP' for row in file_rows):,}",
        f"- Files proposed for deletion: {len(deleted_files):,}",
        f"- Uncertain files retained: {len(uncertain_files):,}",
        f"- Total proposed file size: {sum(int(row['size']) for row in deleted_files):,} bytes",
        "",
        "## Final-pipeline dependency roots",
        "",
        "- `outputs/window_features.csv` and `outputs/feature_schema.json`",
        "- `outputs/minmax_10fold_trial_grouped/`",
        "- `saved_models/minmax_10fold_trial_grouped/`",
        "- `scripts/run_minmax_10fold.py` → `src/minmax_cv.py`, `src/models.py`, `src/utils.py`",
        "- `scripts/generate_xgboost_shap.py`",
        "- `EEG Deception Results.ipynb` and its referenced result/SHAP artifacts",
        "- Original EEG dataset, MATLAB files, paper, Questions PDF, metadata, and processed feature caches",
        "",
        "## Explicit deletion roots",
        "",
        *[f"- `{path}`" for path in roots],
        "",
        "## Complete proposed deletion paths",
        "",
        *[
            f"- `{row['path']}` — {row['reason']}"
            for row in rows
            if row["classification"] == "DELETE"
        ],
        "",
        "## Complete uncertain paths (retained)",
        "",
        *[
            f"- `{row['path']}` — {row['reason']}"
            for row in rows
            if row["classification"] == "UNCERTAIN"
        ],
        "",
        "## Safety result",
        "",
        "Protected-path guard: **PASSED**.",
        "",
        "The deletion list contains no final notebook, feature CSV, research paper, "
        "dataset/session/MATLAB file, final MinMax result, final fold model, SHAP output, "
        "final MinMax/SHAP script, README, requirements file, or processed feature cache.",
        "",
    ]
    MANIFEST_MD.write_text("\n".join(lines), encoding="utf-8")


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_CSV.is_file():
        raise FileNotFoundError("cleanup_manifest.csv must be created and reviewed first")
    with MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def create_backups() -> None:
    rows = read_manifest()
    protected_guard(rows)
    if BACKUP_ZIP.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {BACKUP_ZIP}")
    if NOTEBOOK_BACKUP.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing notebook backup: {NOTEBOOK_BACKUP}"
        )
    manifest_digest = sha256(MANIFEST_CSV)
    delete_entry_rows = [
        row for row in rows if row["classification"] == "DELETE"
    ]
    delete_entry_paths = [
        validate_manifest_relative(row["path"]) for row in delete_entry_rows
    ]
    missing = [
        normalized_relative(path)
        for path in delete_entry_paths
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"DELETE entries missing before backup: {missing}")
    delete_pairs = [
        (row, path)
        for row, path in zip(delete_entry_rows, delete_entry_paths, strict=True)
        if path.is_file()
    ]
    delete_rows = [row for row, _ in delete_pairs]
    delete_files = [path for _, path in delete_pairs]
    size_changes = [
        row["path"]
        for row, path in zip(delete_rows, delete_files, strict=True)
        if path.stat().st_size != int(row["size"])
    ]
    if size_changes:
        raise RuntimeError(
            f"DELETE files changed size after manifest creation: {size_changes}"
        )
    hash_changes = [
        row["path"]
        for row, path in zip(delete_rows, delete_files, strict=True)
        if not row.get("sha256") or sha256(path) != row["sha256"]
    ]
    if hash_changes:
        raise RuntimeError(
            f"DELETE files changed hash after manifest creation: {hash_changes}"
        )
    lowered = [row["path"].lower() for row in delete_rows]
    if len(lowered) != len(set(lowered)):
        raise RuntimeError("Case-insensitive duplicate deletion paths in manifest")

    expected_entries = {str(row["path"]) for row in delete_entry_rows}

    def reconcile_live_deletion_roots() -> None:
        observed_entries: set[str] = set()
        for relative in deletion_roots(rows):
            root = validate_manifest_relative(relative)
            if not root.exists():
                raise RuntimeError(
                    f"Deletion root disappeared before backup: {relative}"
                )
            candidates = [root, *root.rglob("*")] if root.is_dir() else [root]
            for candidate in candidates:
                attributes = getattr(candidate.lstat(), "st_file_attributes", 0)
                if candidate.is_symlink() or attributes & 0x400:
                    raise RuntimeError(
                        "Deletion root contains a link/reparse point: "
                        f"{normalized_relative(candidate)}"
                    )
                observed_entries.add(normalized_relative(candidate))
        if observed_entries != expected_entries:
            raise RuntimeError(
                "Deletion roots changed after manifest creation; "
                f"unmanifested={sorted(observed_entries - expected_entries)}, "
                f"missing={sorted(expected_entries - observed_entries)}"
            )
        changed = [
            row["path"]
            for row, path in delete_pairs
            if path.stat().st_size != int(row["size"])
            or sha256(path) != row["sha256"]
        ]
        if changed:
            raise RuntimeError(
                f"DELETE files changed after manifest creation: {changed}"
            )

    reconcile_live_deletion_roots()

    expected = {
        normalized_relative(path): {
            "size": path.stat().st_size,
            "sha256": row["sha256"],
        }
        for row, path in zip(delete_rows, delete_files, strict=True)
    }
    temporary = BACKUP_ZIP.with_suffix(".zip.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary backup: {temporary}")
    notebook_temporary = NOTEBOOK_BACKUP.with_suffix(".ipynb.tmp")
    if notebook_temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary notebook backup: {notebook_temporary}"
        )
    shutil.copy2(NOTEBOOK, notebook_temporary)
    if sha256(NOTEBOOK) != sha256(notebook_temporary):
        raise RuntimeError("Temporary notebook backup hash does not match")

    rollback_metadata = {
        "__cleanup__/cleanup_manifest.csv": MANIFEST_CSV,
        "__cleanup__/cleanup_manifest.md": MANIFEST_MD,
    }
    archive_expected = dict(expected)
    archive_expected.update(
        {
            relative: {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for relative, path in rollback_metadata.items()
        }
    )
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for relative in sorted(expected):
            archive.write(PROJECT_ROOT / relative, arcname=relative)
        for relative, path in sorted(rollback_metadata.items()):
            archive.write(path, arcname=relative)
    with zipfile.ZipFile(temporary, mode="r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Backup archive failed CRC validation at {corrupt}")
        observed = {
            item.filename: item.file_size
            for item in archive.infolist()
            if not item.is_dir()
        }
        expected_sizes = {
            relative: int(metadata["size"])
            for relative, metadata in archive_expected.items()
        }
        if observed != expected_sizes:
            raise RuntimeError("Backup archive contents or uncompressed sizes differ")
        for relative, metadata in archive_expected.items():
            digest = hashlib.sha256()
            with archive.open(relative, "r") as archived:
                for block in iter(lambda: archived.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != metadata["sha256"]:
                raise RuntimeError(
                    f"Backup SHA-256 differs from source for {relative}"
                )
    reconcile_live_deletion_roots()
    notebook_temporary.replace(NOTEBOOK_BACKUP)
    temporary.replace(BACKUP_ZIP)
    if sha256(MANIFEST_CSV) != manifest_digest:
        raise RuntimeError("Cleanup manifest changed while backup was being created")
    if sha256(NOTEBOOK) != sha256(NOTEBOOK_BACKUP):
        raise RuntimeError("Final notebook pre-cleanup backup hash does not match")
    print(f"Backup archive verified: {BACKUP_ZIP.name}")
    print(f"Deletion files archived: {len(expected):,}")
    print(f"Rollback metadata entries archived: {len(rollback_metadata):,}")
    print(
        "Archived uncompressed bytes: "
        f"{sum(int(metadata['size']) for metadata in expected.values()):,}"
    )
    print(f"Manifest SHA-256: {manifest_digest}")
    print(f"Backup ZIP SHA-256: {sha256(BACKUP_ZIP)}")
    print(f"Notebook backup verified: {NOTEBOOK_BACKUP.name}")


def inventory() -> None:
    rows = scan_entries()
    protected_guard(rows)
    write_manifest(rows)
    file_rows = [
        row
        for row in rows
        if "directory" not in str(row["type"]).lower()
        and row["type"] != "Planned cleanup artifact"
    ]
    deleted = [row for row in file_rows if row["classification"] == "DELETE"]
    uncertain = [row for row in file_rows if row["classification"] == "UNCERTAIN"]
    print("CLEANUP DRY RUN — NO FILES MODIFIED OR DELETED")
    print(f"Files to keep: {sum(row['classification'] == 'KEEP' for row in file_rows):,}")
    print(f"Files proposed for deletion: {len(deleted):,}")
    print(f"Uncertain files retained: {len(uncertain):,}")
    print(f"Total proposed size: {sum(int(row['size']) for row in deleted):,} bytes")
    print("Explicit deletion roots:")
    for relative in deletion_roots(rows):
        print(f"  DELETE {relative}")
    print("Complete proposed deletion paths:")
    for row in rows:
        if row["classification"] == "DELETE":
            print(f"  DELETE {row['path']}")
    print("Complete uncertain file paths:")
    for row in uncertain:
        print(f"  UNCERTAIN {row['path']}")
    print(f"Manifest CSV: {MANIFEST_CSV.name}")
    print(f"Manifest Markdown: {MANIFEST_MD.name}")
    print("Protected-path guard: PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create-backups",
        action="store_true",
        help="Create and verify the deletion ZIP and pre-cleanup notebook backup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.create_backups:
        create_backups()
    else:
        inventory()
