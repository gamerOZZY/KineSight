"""Reproducible command-line pipeline from Silver frames to Gold features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from src.features.biomechanics import (
    ANGLE_FEATURE_NAMES,
    DEVIATION_FEATURE_NAMES,
    HOLLOW_BODY_FEATURE_NAMES,
    PLANK_FEATURE_NAMES,
    extract_frame_features,
    feature_dictionary,
)
from src.features.temporal import (
    MIN_VALID_FRAMES,
    RESAMPLE_TOLERANCE_MS,
    TARGET_FPS,
    WINDOW_SECONDS,
    build_temporal_windows,
)
from src.pose.landmarks import extract_exercise_landmarks
from src.preprocessing.quality_control import quality_frame, write_quality_parquet

LOGGER = logging.getLogger("kinesight.pipeline")
EXERCISES = ("plank", "hollow_body")
GOLD_METADATA_COLUMNS = (
    "video_id",
    "participant_id",
    "exercise",
    "camera_view",
    "source_video",
    "label",
    "error_type",
    "window_start_ms",
    "window_end_ms",
    "frame_ids",
    "n_sampled_frames",
    "n_valid_pose_frames",
)
GOLD_SCHEMA_METADATA = [
    ("video_id", pa.string()),
    ("participant_id", pa.string()),
    ("exercise", pa.string()),
    ("camera_view", pa.string()),
    ("source_video", pa.string()),
    ("label", pa.string()),
    ("error_type", pa.string()),
    ("window_start_ms", pa.int64()),
    ("window_end_ms", pa.int64()),
    ("frame_ids", pa.list_(pa.string())),
    ("n_sampled_frames", pa.int64()),
    ("n_valid_pose_frames", pa.int64()),
]


def _feature_names(exercise: str) -> tuple[str, ...]:
    if exercise == "plank":
        return PLANK_FEATURE_NAMES
    if exercise == "hollow_body":
        return HOLLOW_BODY_FEATURE_NAMES
    raise ValueError(f"Ejercicio no soportado: {exercise}")


def _gold_columns(exercise: str) -> list[str]:
    columns = list(GOLD_METADATA_COLUMNS)
    for feature in _feature_names(exercise):
        columns.append(f"{feature}_valid_frames")
        for statistic in ("mean", "std", "min", "max", "range", "slope"):
            columns.append(f"{feature}_{statistic}")
    return columns


def _gold_schema(exercise: str) -> pa.Schema:
    columns = list(GOLD_SCHEMA_METADATA)
    for feature in _feature_names(exercise):
        columns.append((f"{feature}_valid_frames", pa.int64()))
        columns.extend(
            (f"{feature}_{statistic}", pa.float64())
            for statistic in ("mean", "std", "min", "max", "range", "slope")
        )
    return pa.schema(columns)


def _write_quality(path: Path, observations: list[dict[str, Any]]) -> None:
    write_quality_parquet(path, observations)


def _empty_gold_frame(exercise: str) -> pd.DataFrame:
    return pd.DataFrame(columns=_gold_columns(exercise))


def run_feature_engineering(
    project_root: Path,
    exercise: str,
    *,
    window_seconds: float = WINDOW_SECONDS,
    target_fps: float = TARGET_FPS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create frame-level feature QC, temporal Gold Parquet, and feature dictionary."""
    exercise = exercise.lower()
    feature_names = _feature_names(exercise)
    silver_root = project_root / "data" / "silver" / "landmarks" / exercise
    gold_path = project_root / "data" / "gold" / f"{exercise}_features.parquet"
    dictionary_path = project_root / "data" / "gold" / f"{exercise}_feature_dictionary.json"
    qc_path = silver_root / "feature_quality_observations.parquet"
    output_paths = (gold_path, dictionary_path, qc_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Ya existen resultados Gold/QC; usa overwrite=True para reemplazarlos: "
            + ", ".join(str(path) for path in existing)
        )

    frame_rows = []
    feature_observations = []
    if silver_root.is_dir():
        for silver_path in sorted(silver_root.glob("P*.parquet")):
            landmarks = pd.read_parquet(silver_path, engine="pyarrow")
            for frame in landmarks.to_dict(orient="records"):
                features, invalid = extract_frame_features(frame, exercise)
                frame_feature_row = {
                    key: frame.get(key)
                    for key in (
                        "video_id",
                        "participant_id",
                        "exercise",
                        "camera_view",
                        "source_video",
                        "label",
                        "error_type",
                        "frame_id",
                        "timestamp_ms",
                        "pose_detected",
                    )
                }
                frame_feature_row.update(features)
                frame_rows.append(frame_feature_row)
                feature_observations.extend(
                    quality_frame(
                        frame.get("video_id"),
                        frame.get("frame_id"),
                        frame.get("timestamp_ms"),
                        observation,
                    )
                    for observation in invalid
                )

    frame_columns = [
        "video_id",
        "participant_id",
        "exercise",
        "camera_view",
        "source_video",
        "label",
        "error_type",
        "frame_id",
        "timestamp_ms",
        "pose_detected",
        *feature_names,
    ]
    frame_features = pd.DataFrame(frame_rows, columns=frame_columns)
    if frame_features.empty:
        windows = _empty_gold_frame(exercise)
        window_invalid = []
        LOGGER.info("No hay landmarks de %s; se escribirá un Gold vacío con esquema.", exercise)
    else:
        windows, window_invalid = build_temporal_windows(
            frame_features,
            exercise,
            window_seconds=window_seconds,
            target_fps=target_fps,
            min_valid_frames=min_valid_frames,
        )
        if windows.empty:
            windows = _empty_gold_frame(exercise)
        else:
            for column in _gold_columns(exercise):
                if column not in windows:
                    windows[column] = None
            windows = windows[_gold_columns(exercise)]

    feature_observations.extend(window_invalid)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_table = pa.Table.from_pandas(
        windows,
        schema=_gold_schema(exercise),
        preserve_index=False,
    )
    pq.write_table(gold_table, gold_path)
    dictionary = feature_dictionary(exercise)
    dictionary["temporal_aggregation"] = {
        "window_seconds": window_seconds,
        "target_fps": target_fps,
        "minimum_valid_frames_per_feature": min_valid_frames,
        "resample_tolerance_ms": RESAMPLE_TOLERANCE_MS,
        "interpolation": "none",
        "statistics": {
            "mean": {
                "definition": "Media aritmética de los valores válidos de la característica.",
                "formula": "sum(values) / count(values)",
                "units": "same_as_feature",
            },
            "std": {
                "definition": "Desviación estándar poblacional de los valores válidos.",
                "formula": "sqrt(mean((value - mean(values)) ** 2))",
                "units": "same_as_feature",
            },
            "min": {
                "definition": "Valor mínimo válido en la ventana.",
                "formula": "min(values)",
                "units": "same_as_feature",
            },
            "max": {
                "definition": "Valor máximo válido en la ventana.",
                "formula": "max(values)",
                "units": "same_as_feature",
            },
            "range": {
                "definition": "Diferencia entre el máximo y el mínimo válidos.",
                "formula": "max(values) - min(values)",
                "units": "same_as_feature",
            },
            "slope": {
                "definition": "Pendiente OLS de la característica respecto al tiempo.",
                "formula": "cov(time_seconds, values) / var(time_seconds)",
                "units": "feature_units_per_second",
            },
        },
    }
    dictionary_path.write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_quality(qc_path, feature_observations)
    LOGGER.info(
        "%s: %d filas Gold, %d observaciones QC; salida %s",
        exercise,
        len(windows),
        len(feature_observations),
        gold_path,
    )
    return {
        "exercise": exercise,
        "gold_rows": len(windows),
        "frame_rows": len(frame_features),
        "quality_observations": len(feature_observations),
        "gold_path": gold_path,
        "feature_dictionary_path": dictionary_path,
        "window_seconds": window_seconds,
        "target_fps": target_fps,
        "minimum_valid_frames": min_valid_frames,
        "angle_features": sorted(set(feature_names) & ANGLE_FEATURE_NAMES),
        "deviation_features": sorted(set(feature_names) & DEVIATION_FEATURE_NAMES),
    }


def run_pipeline(
    project_root: Path,
    exercise: str,
    model_path: Path,
    *,
    sampled_fps: float = 7.0,
    overwrite: bool = False,
    window_seconds: float = WINDOW_SECONDS,
    target_fps: float = TARGET_FPS,
    min_valid_frames: int = MIN_VALID_FRAMES,
) -> dict[str, Any]:
    """Run landmark extraction and temporal feature engineering for one exercise."""
    landmark_result = extract_exercise_landmarks(
        project_root,
        exercise,
        model_path,
        sampled_fps=sampled_fps,
        overwrite=overwrite,
    )
    feature_result = run_feature_engineering(
        project_root,
        exercise,
        window_seconds=window_seconds,
        target_fps=target_fps,
        min_valid_frames=min_valid_frames,
        overwrite=overwrite,
    )
    return {"landmarks": landmark_result, "features": feature_result}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", choices=EXERCISES, required=True)
    parser.add_argument(
        "--phase",
        choices=("landmarks", "features", "all"),
        default="all",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose/pose_landmarker.task"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sampled-fps", type=float, default=7.0)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--min-valid-frames", type=int, default=MIN_VALID_FRAMES)
    args = parser.parse_args()
    _configure_logging()

    project_root = args.project_root.resolve()
    model_path = args.model
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if args.phase in {"landmarks", "all"}:
        result = extract_exercise_landmarks(
            project_root,
            args.exercise,
            model_path,
            sampled_fps=args.sampled_fps,
            overwrite=args.overwrite,
        )
        LOGGER.info("Landmarks: %s", result)
    if args.phase in {"features", "all"}:
        result = run_feature_engineering(
            project_root,
            args.exercise,
            window_seconds=args.window_seconds,
            target_fps=args.target_fps,
            min_valid_frames=args.min_valid_frames,
            overwrite=args.overwrite,
        )
        LOGGER.info("Features: %s", result)


if __name__ == "__main__":
    main()
