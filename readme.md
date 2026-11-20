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

Run the phases from the repository root:

```powershell
python -m src.pipeline --exercise plank --phase landmarks
python -m src.pipeline --exercise plank --phase features
python -m src.pipeline --exercise hollow_body --phase all
```

The equivalent notebook entry points are `notebooks/02_silver_landmarks.ipynb`
and `notebooks/03_gold_features.ipynb`. Landmark Parquet files are written per
video under `data/silver/landmarks/`; frame-level and feature-level quality
observations are stored alongside them. Gold windows are written to
`data/gold/plank_features.parquet` and
`data/gold/hollow_body_features.parquet`, with a feature dictionary JSON for
each exercise.

Silver extraction uses the original video FPS and the 7 FPS sampling schedule
from the frame-extraction notebook to reconstruct source frame indices and
millisecond timestamps. This reconstruction assumes constant-frame-rate source
videos. The feature notebook uses non-overlapping 2-second windows, a 5 FPS
target grid, and requires at least five valid observations per feature and
window. Missing landmarks are not interpolated.
