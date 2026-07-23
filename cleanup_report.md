# Project Cleanup Report

## Completion

- Cleanup completed: `2026-07-23T20:52:10+04:00`
- Project root: `C:\Users\ADMIN\Desktop\EEG Deception Project`
- Final result: obsolete experiment artifacts removed; final MinMax trial-grouped
  10-fold pipeline retained and validated
- Models retrained during cleanup: **No**
- Features, cross-validation, predictions, and SHAP recomputed during cleanup:
  **No**

## Preserved material

- Original dataset: `Dataset_32302035/`
  - 952 non-empty original `.mat` EEG files
  - all session folders
  - `Questions.pdf`
- Research paper:
  `A Multi-Class EEG Dataset for Behavioral Roles in Deception Honesty, Bluffing, Lying, and Deceiving.pdf`
- Final feature and reconstruction data:
  - `outputs/window_features.csv`
  - `outputs/feature_schema.json`
  - `outputs/dataset_metadata.csv`
  - `outputs/dataset_summary.json`
  - `outputs/preprocessing_qc.csv`
  - all seven files under `cache/`
- Final result directory:
  `outputs/minmax_10fold_trial_grouped/`
- Final saved-model directory:
  `saved_models/minmax_10fold_trial_grouped/`
  - 80 fold pipelines in total
  - 10 XGBoost fold pipelines
- Final notebook: `EEG Deception Results.ipynb`
- Notebook backups:
  - `EEG Deception Results.backup.ipynb`
  - `EEG Deception Results.pre_cleanup_backup.ipynb`
- Final and shared source code, tests, requirements, environments, and project
  configuration
- Cleanup provenance:
  - `cleanup_manifest.csv`
  - `cleanup_manifest.md`
  - `scripts/prepare_cleanup.py`

No final MinMax result, SHAP artifact, final fold model, dataset file, research
paper, feature input, or notebook source/output cell was deleted.

## Deleted material

The following original project entries were classified `DELETE`, backed up,
moved to `_cleanup_archive`, validated there, and then removed:

| Category | Files | Bytes |
|---|---:|---:|
| Aborted partial MinMax preflight | 39 | 196,132,780 |
| Obsolete root subject-wise outputs | 38 | 5,090,435 |
| Obsolete root subject-wise models | 8 | 45,247,997 |
| Pytest cache | 5 | 2,067 |
| Python bytecode caches | 21 | 298,297 |
| **Total** | **111** | **246,771,576** |

Fourteen original folders were removed:

- 8 folders under `partial_runs/`
- 3 folders under `.pytest_cache/`
- `scripts/__pycache__/`
- `src/__pycache__/`
- `tests/__pycache__/`

The temporary archive also required five non-original parent containers to
preserve relative paths. After verification, 19 empty descendant directories
and the empty `_cleanup_archive` root were removed individually.

## Space and rollback

- Obsolete bytes removed from the active project tree:
  **246,771,576 bytes (235.34 MiB)**
- Verified compressed rollback ZIP:
  `C:\Users\ADMIN\Desktop\EEG Deception Project\project_cleanup_backup.zip`
- ZIP size: **85,095,248 bytes (81.15 MiB)**
- Active-tree reduction after retaining the ZIP:
  **161,676,328 bytes (154.19 MiB)**
- ZIP SHA-256:
  `5cd6892b2018836709d9dabdafc84ab3aa40ee34b0ad3f5202d74b2676da097f`

The ZIP contains all 111 deleted files plus embedded copies of
`cleanup_manifest.csv` and `cleanup_manifest.md`. Its entry names, sizes, CRCs,
and SHA-256 values were checked against the live sources before deletion. It
does not contain the EEG dataset, virtual environments, or large processed
caches.

The pre-cleanup notebook backup was hash-identical when it was created. VS Code
later changed only the live notebook kernelspec label; all 43 cell sources,
execution counts, and 20 stored outputs remained identical. Both files were
preserved.

## Uncertain files retained

The following four files were classified `UNCERTAIN` and were not deleted:

1. `outputs/minmax_10fold_trial_grouped/comparison_with_subjectwise.csv`
   Historical comparison retained because the final reproducibility manifest
   hashes it. The final notebook does not load or display it.
2. `scripts/run_ml_experiment.py`
   Legacy runner retained as provenance for the preserved feature-generation
   workflow and shared imports.
3. `src/evaluation.py`
   Shared legacy evaluation functions retained because contract tests import
   them.
4. `tests/test_pipeline_contracts.py`
   Mixed shared/legacy tests retained because they validate preserved loading,
   preprocessing, feature, model, and aggregation behavior.

## Post-cleanup validation

All required gates passed:

- `outputs/window_features.csv` is readable.
- All 15 final result CSV files are readable and non-empty.
- All 13 final figures have valid PNG or PDF signatures.
- The final reproducibility manifest has 103/103 present, matching
  `artifact_sha256` entries.
- Exactly 80 final fold models remain, 10 per classifier.
- All 10 XGBoost fold pipelines load successfully with 496 ordered features,
  classes `[0, 1, 2, 3]`, and pipeline steps `imputer`, `scaler`, and
  `classifier`.
- Core MinMax imports pass in Python 3.10.
- SHAP/XGBoost imports pass in Python 3.13 with SHAP 0.52.0 and XGBoost 3.2.0.
- Test suite result: **16 passed**.
- The notebook strictly validates as nbformat 4.5 with 43 cells, 18 code
  cells, execution counts 1–18, 20 outputs, and zero stored errors.
- A temporary execution of the notebook completed with zero errors while
  loading retained outputs only; the temporary file was removed afterward.
- All 23 directly referenced notebook result/SHAP files exist and are
  non-empty.
- No obsolete root subject-wise result, root model, or `StandardScaler`
  result is loaded or displayed by the notebook.
- The final reproduction commands point to existing scripts and inputs.
- `EEG Deception Results.ipynb`, `outputs/window_features.csv`, the final
  result directory, and all final XGBoost models exist.

## Unresolved provenance note

The retained final `reproducibility_manifest.json` includes a historical
`previous_subjectwise_artifact_sha256` section. Forty-six paths in that
historical-only section refer to the obsolete root outputs/models removed by
this cleanup. Those original bytes remain recoverable from
`project_cleanup_backup.zip`.

This does not affect the active final artifact set: all 103 paths in the
manifest's final `artifact_sha256` section remain present and hash-valid. A
future rerun of the MinMax script will omit its optional historical comparison
when those root subject-wise files are absent.

## Final paths

- Results:
  `C:\Users\ADMIN\Desktop\EEG Deception Project\outputs\minmax_10fold_trial_grouped`
- Saved models:
  `C:\Users\ADMIN\Desktop\EEG Deception Project\saved_models\minmax_10fold_trial_grouped`
- Notebook:
  `C:\Users\ADMIN\Desktop\EEG Deception Project\EEG Deception Results.ipynb`
- Backup:
  `C:\Users\ADMIN\Desktop\EEG Deception Project\project_cleanup_backup.zip`
- Manifest:
  `C:\Users\ADMIN\Desktop\EEG Deception Project\cleanup_manifest.csv`
