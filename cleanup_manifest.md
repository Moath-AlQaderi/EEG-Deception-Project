# Project Cleanup Dry-Run Manifest

Generated: 2026-07-23T20:20:46.488184+04:00

No files were moved or deleted while creating this manifest.

## Counts

- Files to keep: 20,010
- Files proposed for deletion: 111
- Uncertain files retained: 4
- Total proposed file size: 246,771,576 bytes

## Final-pipeline dependency roots

- `outputs/window_features.csv` and `outputs/feature_schema.json`
- `outputs/minmax_10fold_trial_grouped/`
- `saved_models/minmax_10fold_trial_grouped/`
- `scripts/run_minmax_10fold.py` → `src/minmax_cv.py`, `src/models.py`, `src/utils.py`
- `scripts/generate_xgboost_shap.py`
- `EEG Deception Results.ipynb` and its referenced result/SHAP artifacts
- Original EEG dataset, MATLAB files, paper, Questions PDF, metadata, and processed feature caches

## Explicit deletion roots

- `.pytest_cache`
- `partial_runs`
- `outputs/artifact_verification.json`
- `outputs/classification_report_decision_tree.txt`
- `outputs/classification_report_knn.txt`
- `outputs/classification_report_logistic_regression.txt`
- `outputs/classification_report_mlp.txt`
- `outputs/classification_report_naive_bayes.txt`
- `outputs/classification_report_random_forest.txt`
- `outputs/classification_report_svm.txt`
- `outputs/classification_report_xgboost.txt`
- `outputs/confusion_matrix_decision_tree.png`
- `outputs/confusion_matrix_knn.png`
- `outputs/confusion_matrix_logistic_regression.png`
- `outputs/confusion_matrix_mlp.png`
- `outputs/confusion_matrix_naive_bayes.png`
- `outputs/confusion_matrix_random_forest.png`
- `outputs/confusion_matrix_svm.png`
- `outputs/confusion_matrix_xgboost.png`
- `outputs/data_split.csv`
- `outputs/final_summary.txt`
- `outputs/leakage_checks.json`
- `outputs/metrics.json`
- `outputs/model_comparison.csv`
- `outputs/model_comparison.png`
- `outputs/per_class_trial_metrics.csv`
- `outputs/reproducibility_manifest.json`
- `outputs/run.log`
- `outputs/split_assignments.csv`
- `outputs/split_summary.json`
- `outputs/top_features_decision_tree.csv`
- `outputs/top_features_decision_tree.png`
- `outputs/top_features_random_forest.csv`
- `outputs/top_features_random_forest.png`
- `outputs/top_features_xgboost.csv`
- `outputs/top_features_xgboost.png`
- `outputs/trial_level_metrics.csv`
- `outputs/trial_predictions.csv`
- `outputs/window_level_metrics.csv`
- `outputs/window_predictions.csv`
- `saved_models/decision_tree.joblib`
- `saved_models/knn.joblib`
- `saved_models/logistic_regression.joblib`
- `saved_models/mlp.joblib`
- `saved_models/naive_bayes.joblib`
- `saved_models/random_forest.joblib`
- `saved_models/svm.joblib`
- `saved_models/xgboost.joblib`
- `scripts/__pycache__`
- `src/__pycache__`
- `tests/__pycache__`

## Complete proposed deletion paths

- `.pytest_cache` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/.gitignore` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/CACHEDIR.TAG` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/README.md` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/v` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/v/cache` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/v/cache/lastfailed` — Generated test or Python bytecode cache; safely recreated.
- `.pytest_cache/v/cache/nodeids` — Generated test or Python bytecode cache; safely recreated.
- `outputs/artifact_verification.json` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_decision_tree.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_knn.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_logistic_regression.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_mlp.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_naive_bayes.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_random_forest.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_svm.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/classification_report_xgboost.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_decision_tree.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_knn.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_logistic_regression.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_mlp.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_naive_bayes.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_random_forest.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_svm.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/confusion_matrix_xgboost.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/data_split.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/final_summary.txt` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/leakage_checks.json` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/metrics.json` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/model_comparison.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/model_comparison.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/per_class_trial_metrics.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/reproducibility_manifest.json` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/run.log` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/split_assignments.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/split_summary.json` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/top_features_decision_tree.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/top_features_decision_tree.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/top_features_random_forest.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/top_features_random_forest.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/top_features_xgboost.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/top_features_xgboost.png` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/trial_level_metrics.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/trial_predictions.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/window_level_metrics.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `outputs/window_predictions.csv` — Root-level artifact from the obsolete subject-wise 80/20 holdout; the final notebook and SHAP workflow load only the MinMax subdirectory.
- `partial_runs` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/outputs` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/outputs/run.log` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_01.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_02.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_03.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_04.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_05.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_06.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_07.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_08.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_09.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/decision_tree/fold_10.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_01.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_02.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_03.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_04.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_05.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_06.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_07.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_08.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_09.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/logistic_regression/fold_10.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_01.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_02.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_03.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_04.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_05.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_06.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_07.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_08.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_09.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/random_forest/fold_10.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_01.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_02.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_03.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_04.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_05.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_06.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_07.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `partial_runs/aborted_minmax_preflight_20260722/saved_models/xgboost/fold_08.joblib` — Explicitly aborted MinMax preflight containing incomplete duplicate models and a truncated log.
- `saved_models/decision_tree.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/knn.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/logistic_regression.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/mlp.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/naive_bayes.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/random_forest.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/svm.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `saved_models/xgboost.joblib` — Single holdout model from the obsolete subject-wise pipeline; superseded by the 10 final fold-specific models for this classifier.
- `scripts/__pycache__` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/__init__.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/__init__.cpython-313.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/extract_features.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/generate_xgboost_shap.cpython-313.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/inspect_dataset.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/prepare_cleanup.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/run_minmax_10fold.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `scripts/__pycache__/run_ml_experiment.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/__init__.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/data_loader.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/evaluation.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/feature_extraction.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/minmax_cv.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/models.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/preprocessing.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/utils.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `src/__pycache__/windowing.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `tests/__pycache__` — Generated test or Python bytecode cache; safely recreated.
- `tests/__pycache__/test_minmax_cv.cpython-310-pytest-9.1.1.pyc` — Generated test or Python bytecode cache; safely recreated.
- `tests/__pycache__/test_minmax_cv.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.
- `tests/__pycache__/test_pipeline_contracts.cpython-310-pytest-9.1.1.pyc` — Generated test or Python bytecode cache; safely recreated.
- `tests/__pycache__/test_pipeline_contracts.cpython-310.pyc` — Generated test or Python bytecode cache; safely recreated.

## Complete uncertain paths (retained)

- `outputs/minmax_10fold_trial_grouped/comparison_with_subjectwise.csv` — Historical comparison is not displayed by the final notebook, but it is hashed by the preserved final MinMax reproducibility manifest.
- `scripts/run_ml_experiment.py` — Legacy holdout runner, retained because it imports shared feature-generation modules and documents how the preserved feature table was originally produced.
- `src/evaluation.py` — Legacy evaluation module, retained because a preserved contract test imports its trial-probability aggregation function.
- `tests/test_pipeline_contracts.py` — Mixed legacy/shared contract tests; retained because most tests validate the preserved preprocessing, windowing, feature, and model components.

## Safety result

Protected-path guard: **PASSED**.

The deletion list contains no final notebook, feature CSV, research paper, dataset/session/MATLAB file, final MinMax result, final fold model, SHAP output, final MinMax/SHAP script, README, requirements file, or processed feature cache.
