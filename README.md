# EEG Deception Classification: Final MinMax 10-Fold Experiment

This repository contains the final classical machine-learning experiment for
classifying four EEG behavioral roles: Honest, Bluffer, Liar, and Deceiver.
The retained result is a non-subject-wise, trial-grouped 10-fold
cross-validation experiment using fold-local `MinMaxScaler`, trial-level
probability aggregation, and post-hoc XGBoost SHAP explanations.

The conference-ready results are displayed in
`EEG Deception Results.ipynb`. No deep-learning model, hyperparameter search,
or model-selection retraining is part of this experiment.

## Data and feature input

The raw dataset is not committed to this Git repository because the local copy
is approximately 9.3 GB. Download it from the authors' Figshare record:

- Dataset (latest record): [Figshare DOI 10.6084/m9.figshare.32302035](https://doi.org/10.6084/m9.figshare.32302035)
- Dataset used here (version 1): [Figshare DOI 10.6084/m9.figshare.32302035.v1](https://doi.org/10.6084/m9.figshare.32302035.v1)
- Paper: [Data 2026, 11(7), 162](https://doi.org/10.3390/data11070162)
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Place the downloaded folder at `Dataset_32302035/` to reproduce the local
layout used by the preprocessing code. Please cite both the dataset and its
data descriptor:

> Alazrai, R.; Alzaatreh, L. (2026). “An EEG Dataset for Behavioral Roles in
> Deception: Honesty, Bluffing, Lying, and Deceiving.” *figshare*. Dataset,
> version 1. https://doi.org/10.6084/m9.figshare.32302035.v1

> Alzaatreh, L. M.; Hatahet, O.; Alazrai, R. “A Multi-Class EEG Dataset for
> Behavioral Roles in Deception: Honesty, Bluffing, Lying, and Deceiving.”
> *Data* **2026**, *11*(7), 162.
> https://doi.org/10.3390/data11070162

The original research material contains:

- 952 original MATLAB EEG trial files from 17 sessions
- 51 original participants and four behavioral roles
- 16 EEG channels sampled at 2048 Hz
- the research paper and `Dataset_32302035/Questions.pdf`
- metadata and processed caches used to create the final feature table

The final experiment begins with `outputs/window_features.csv`. After
response-free preprocessing, severe-artifact QC, five-second thinking-epoch
extraction, resampling to 256 Hz, one-second windows with 50% overlap, and
channel-wise feature extraction, the retained table contains 7,200 windows
from 800 trials, 48 participants, and all 17 sessions. Each window has 496
ordered EEG features.

Identifiers, paths, role labels, question IDs, and window indices are retained
for grouping and auditing but are excluded from the model feature matrix.

## Final experiment

`StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` uses
`trial_id` as the grouping variable. All overlapping windows from a trial stay
in one fold, and every retained window receives exactly one out-of-fold
prediction.

Each fold-specific pipeline contains:

1. `SimpleImputer(strategy="median", keep_empty_features=True)`
2. `MinMaxScaler(feature_range=(0, 1))`
3. one fixed classifier

The eight classifiers are:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- RBF Support Vector Machine
- distance-weighted K-Nearest Neighbors
- Gaussian Naive Bayes
- scikit-learn MLP `(128, 64)`

No tuning is performed. Trial predictions are primary: the four class
probabilities are averaged across the nine windows belonging to each trial,
then the largest average selects the predicted role. Window metrics are
secondary because overlapping windows from the same trial are correlated.

This evaluation is intentionally non-subject-wise. Trial overlap is zero in
every fold, but participants occur in both training and validation partitions
(45.2 overlapping participants per fold on average). The reported scores
therefore do not demonstrate generalization to unseen participants.

## Final results

XGBoost ranks first by pooled out-of-fold trial macro F1:

| Metric | Value |
|---|---:|
| Trial accuracy | 0.968750 |
| Trial macro F1 | 0.968523 |
| Trial weighted F1 | 0.968748 |
| Trial Cohen's kappa | 0.958317 |
| Window accuracy | 0.956250 |
| Window macro F1 | 0.955930 |

These values are read from the retained result tables; they were not
recomputed during project cleanup.

## XGBoost SHAP

The SHAP stage loads the 10 saved XGBoost fold pipelines and explains each
window only with the model for the fold in which that window was held out.
It covers all 7,200 windows, 496 features, and four classes. Tables include
global and class-wise mean absolute SHAP values, an explicit feature-name
mapping, and additivity checks. The PNG and vector PDF figures are under
`outputs/minmax_10fold_trial_grouped/shap/`.

SHAP importance is associative and must not be interpreted as proof of
brain-region causality.

## Reproduction

Run commands from the project root on Windows. The classical experiment uses
Python 3.10:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_minmax_10fold.py --features outputs\window_features.csv --output-dir outputs\minmax_10fold_trial_grouped --models-dir saved_models\minmax_10fold_trial_grouped --seed 42
```

SHAP 0.52.0 is installed in Python 3.13:

```powershell
py -3.13 -m venv .shap-venv
.\.shap-venv\Scripts\python.exe -m pip install -r requirements.txt
.\.shap-venv\Scripts\python.exe scripts\generate_xgboost_shap.py --features outputs\window_features.csv --results-dir outputs\minmax_10fold_trial_grouped --models-dir saved_models\minmax_10fold_trial_grouped\xgboost --notebook "EEG Deception Results.ipynb" --seed 42
```

Open `EEG Deception Results.ipynb` to display the saved results. Its cells load
existing tables and figures; they do not train models or recompute SHAP.

Run the retained tests without creating project caches:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
```

## Retained project structure

```text
EEG Deception Project/
├── Dataset_32302035/                     # Download separately; ignored by Git
├── cache/                                # Local processed arrays; ignored by Git
├── outputs/
│   ├── window_features.csv               # Final model input
│   ├── feature_schema.json
│   ├── dataset_metadata.csv
│   ├── dataset_summary.json
│   ├── preprocessing_qc.csv
│   └── minmax_10fold_trial_grouped/       # Final metrics, OOF predictions, figures, SHAP
├── saved_models/
│   └── minmax_10fold_trial_grouped/       # 80 pipelines; 10 folds × 8 models
├── scripts/
│   ├── run_minmax_10fold.py
│   ├── generate_xgboost_shap.py
│   └── prepare_cleanup.py
├── src/                                  # Shared loading, model, and CV modules
├── tests/
├── EEG Deception Results.ipynb
├── requirements.txt
├── cleanup_manifest.csv
├── cleanup_manifest.md
├── cleanup_report.md
└── project_cleanup_backup.zip            # Local rollback archive; ignored by Git
```

Key final files include:

- `outputs/minmax_10fold_trial_grouped/model_comparison.csv`
- `outputs/minmax_10fold_trial_grouped/fold_metrics.csv`
- `outputs/minmax_10fold_trial_grouped/pooled_trial_metrics.csv`
- `outputs/minmax_10fold_trial_grouped/oof_window_predictions.csv`
- `outputs/minmax_10fold_trial_grouped/oof_trial_predictions.csv`
- `outputs/minmax_10fold_trial_grouped/reproducibility_manifest.json`
- `saved_models/minmax_10fold_trial_grouped/`

The cleanup manifests and report document which obsolete artifacts were
removed and how their verified rollback ZIP was created.

The 80 saved `.joblib` pipelines are versioned through Git LFS rather than
ordinary Git blobs.
