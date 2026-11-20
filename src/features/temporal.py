"""Timestamp-based resampling and non-overlapping temporal windows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.features.biomechanics import (
    ANGLE_FEATURE_NAMES,
    DEVIATION_FEATURE_NAMES,
    HOLLOW_BODY_FEATURE_NAMES,
    PLANK_FEATURE_NAMES,
)

WINDOW_SECONDS = 2.0
TARGET_FPS = 5.0
MIN_VALID_FRAMES = 5
RESAMPLE_TOLERANCE_MS = 100
STATISTICS = ("mean", "std", "min", "max", "range", "slope")


def _nearest_unique_samples(
    frame_group: pd.DataFrame,
    window_start_ms: int,
    window_end_ms: int,
    target_fps: float,
    tolerance_ms: int,
) -> pd.DataFrame:
    frames = frame_group[
        (frame_group["timestamp_ms"] >= window_start_ms)
        & (frame_group["timestamp_ms"] < window_end_ms)
    ].sort_values("timestamp_ms")
    if frames.empty:
        return frames

    selected_indices: set[Any] = set()
    selected_rows = []
    step_ms = 1000.0 / target_fps
    target_times = np.arange(window_start_ms, window_end_ms, step_ms)
    for target_time in target_times:
        differences = (frames["timestamp_ms"] - target_time).abs()
        candidates = frames.loc[~frames.index.isin(selected_indices)].copy()
        if candidates.empty:
            break
        candidate_differences = (candidates["timestamp_ms"] - target_time).abs()
        closest_index = candidate_differences.idxmin()
        if float(candidate_differences.loc[closest_index]) <= tolerance_ms:
            selected_indices.add(closest_index)
            selected_rows.append(frames.loc[closest_index])
    if not selected_rows:
        return frames.iloc[0:0]
    return pd.DataFrame(selected_rows).sort_values("timestamp_ms")


def _statistics(
    values: Sequence[float],
    timestamps_ms: Sequence[int],
    window_start_ms: int,
) -> dict[str, float]:
    numeric = np.asarray(values, dtype=float)
    times = (np.asarray(timestamps_ms, dtype=float) - window_start_ms) / 1000.0
    centered_times = times - times.mean()
    denominator = float(np.dot(centered_times, centered_times))
    slope = (
        float(np.dot(centered_times, numeric - numeric.mean()) / denominator)
        if denominator > np.finfo(float).eps
        else 0.0
    )
    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "range": float(numeric.max() - numeric.min()),
        "slope": slope,
    }


def build_temporal_windows(
    frame_features: pd.DataFrame,
    exercise: str,
    *,
    window_seconds: float = WINDOW_SECONDS,
    target_fps: float = TARGET_FPS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    tolerance_ms: int = RESAMPLE_TOLERANCE_MS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aggregate feature values into label-homogeneous, fixed-duration windows."""
    if exercise == "plank":
        feature_names = PLANK_FEATURE_NAMES
    elif exercise == "hollow_body":
        feature_names = HOLLOW_BODY_FEATURE_NAMES
    else:
        raise ValueError(f"Ejercicio no soportado: {exercise}")
    required = {
        "video_id", "participant_id", "exercise", "camera_view",
        "timestamp_ms", "frame_id", "label", "error_type", "source_video",
        "pose_detected", *feature_names,
    }
    missing_columns = required - set(frame_features.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas para ventanas temporales: {sorted(missing_columns)}")
    if window_seconds <= 0 or target_fps <= 0 or min_valid_frames < 1:
        raise ValueError("window_seconds, target_fps y min_valid_frames deben ser positivos.")

    window_ms = int(round(window_seconds * 1000))
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    group_columns = ["video_id", "label", "error_type"]
    for _, group in frame_features.groupby(group_columns, dropna=False, sort=False):
        group = group.sort_values("timestamp_ms")
        max_timestamp = int(group["timestamp_ms"].max())
        first_window = int(group["timestamp_ms"].min()) // window_ms * window_ms
        number_of_windows = max(
            1,
            math.floor((max_timestamp - first_window) / window_ms) + 1,
        )
        for window_index in range(number_of_windows):
            start_ms = first_window + window_index * window_ms
            end_ms = start_ms + window_ms
            sampled = _nearest_unique_samples(
                group,
                start_ms,
                end_ms,
                target_fps,
                tolerance_ms,
            )
            if sampled.empty:
                continue
            first = group.iloc[0]
            row: dict[str, Any] = {
                "video_id": first["video_id"],
                "participant_id": first["participant_id"],
                "exercise": first["exercise"],
                "camera_view": first["camera_view"],
                "source_video": first["source_video"],
                "label": first["label"],
                "error_type": first["error_type"],
                "window_start_ms": start_ms,
                "window_end_ms": end_ms,
                "frame_ids": sampled["frame_id"].astype(str).tolist(),
                "n_sampled_frames": int(len(sampled)),
                "n_valid_pose_frames": int(sampled["pose_detected"].fillna(False).astype(bool).sum()),
            }
            for feature_name in feature_names:
                numeric_values = pd.to_numeric(sampled[feature_name], errors="coerce")
                valid_mask = numeric_values.notna() & np.isfinite(numeric_values)
                values = numeric_values[valid_mask].astype(float).tolist()
                timestamps = sampled.loc[valid_mask, "timestamp_ms"].astype(int).tolist()
                row[f"{feature_name}_valid_frames"] = len(values)
                if len(values) < min_valid_frames:
                    for statistic in STATISTICS:
                        row[f"{feature_name}_{statistic}"] = None
                    invalid.append(
                        {
                            "video_id": first["video_id"],
                            "frame_id": None,
                            "timestamp_ms": start_ms,
                            "scope": "window_feature",
                            "landmark_name": None,
                            "field": None,
                            "feature_name": feature_name,
                            "reason": f"insufficient_valid_frames:{len(values)}<{min_valid_frames}",
                            "value": None,
                        }
                    )
                    continue
                row.update(
                    {
                        f"{feature_name}_{name}": value
                        for name, value in _statistics(values, timestamps, start_ms).items()
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), invalid
