# KineSight


## Project Overview
KineSight is an Computer Vision system which detects whether an athlete is well perfoming a physical excercise or not


## Project Objectives


## Table of Contents

<!-- 
## Data architecture
the data is set on a medallion architecture:
Bronze -> 
-->

## Pose and feature pipeline

Install the Python dependencies with `python -m pip install -r requirements.txt`.
Place the MediaPipe Pose Landmarker Lite task file at
`models/pose/pose_landmarker.task` before running the landmark phase.

Run the end-to-end pipeline from the repository root:

```powershell
python -m src.pipeline --exercise plank --phase all
python -m src.pipeline --exercise hollow_body --phase all
```

The `all` phase creates/validates the exercise metadata from Bronze video names,
extracts Silver JPEG frames at 7 FPS if they do not already exist, then runs
MediaPipe landmarks and Gold feature engineering. Existing metadata and complete
frame extractions are validated and preserved; existing Parquet outputs require
the explicit `--overwrite` flag to be replaced. The equivalent notebooks are
`notebooks/02_silver_landmarks.ipynb`, `notebooks/03_gold_features.ipynb`, and
the end-to-end entry point `notebooks/04_bronze_to_gold.ipynb`. Landmark Parquet
files are written per video under `data/silver/landmarks/`; frame-level and
feature-level quality observations are stored alongside them. Gold windows are written to
`data/gold/plank_features.parquet` and
`data/gold/hollow_body_features.parquet`, with a feature dictionary JSON for
each exercise.

Silver extraction uses the original video FPS and the 7 FPS sampling schedule
from the frame-extraction notebook to reconstruct source frame indices and
millisecond timestamps. This reconstruction assumes constant-frame-rate source
videos. The feature notebook uses non-overlapping 2-second windows, a 5 FPS
target grid, and requires at least five valid observations per feature and
window. Missing landmarks are not interpolated.

## Model experiments

The grouped model experiments use only biomechanical window statistics as
predictors; identifiers, class labels, camera metadata, and landmark-validity
counts are excluded. Imputation and scaling are fitted inside each
scikit-learn pipeline. Five-fold nested `StratifiedGroupKFold` keeps every
participant entirely within either the train or validation side of each fold.

```powershell
python -m src.training.train_plank
python -m src.training.train_hollow_body

# Ejecutar los evaluadores experimentales en cámara
python evaluate_plank.py
python evaluate_hollow_body.py
```

The comparison includes Logistic Regression, SVM, Random Forest, Extra Trees,
HistGradientBoosting, and XGBoost. XGBoost is installed with the other project
dependencies (`pip install -r requirements.txt`).

Reports are written under `reports/<exercise>/`; the refitted pipelines and
metadata are saved under `models/<exercise>/`. The current datasets contain
few participants, so the selected model and cross-validation estimates must be
treated as provisional until more independent participants are collected.

The live apps share the same world-landmark biomechanics and 2-second temporal
aggregation used for Gold. They suppress predictions until every model feature
has at least five valid samples, and smooth the latest five window predictions
with a majority vote. No confidence percentage is shown because the selected
models have not been calibrated for deployment. Both interfaces prominently
identify the current models as experimental.
