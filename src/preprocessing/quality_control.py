"""Quality checks and invalid-observation records for pose landmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.features.biomechanics import (
    DEFAULT_PRESENCE_THRESHOLD,
    DEFAULT_VISIBILITY_THRESHOLD,
    LANDMARK_NAMES,
)

LANDMARK_FIELDS = ("x", "y", "z", "visibility", "presence")
QUALITY_COLUMNS = (
    "video_id",
    "frame_id",
    "timestamp_ms",
    "scope",
    "landmark_name",
    "field",
    "feature_name",
    "reason",
    "value",
)
QUALITY_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("frame_id", pa.string()),
        ("timestamp_ms", pa.int64()),
        ("scope", pa.string()),
        ("landmark_name", pa.string()),
        ("field", pa.string()),
        ("feature_name", pa.string()),
        ("reason", pa.string()),
        ("value", pa.string()),
    ]
)


def write_quality_parquet(
    path: Path,
    observations: list[dict[str, Any]],
) -> None:
    """Write QC observations with a stable schema, including when there are none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for observation in observations:
        normalized.append(
            {
                column: (
                    str(observation[column])
                    if column == "value" and observation.get(column) is not None
                    else observation.get(column)
                )
                for column in QUALITY_COLUMNS
            }
        )
    table = pa.Table.from_pylist(normalized, schema=QUALITY_SCHEMA)
    pq.write_table(table, path)


def validate_landmark_frame(frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Record invalid or unavailable landmark values without modifying coordinates."""
    observations: list[dict[str, Any]] = []
    metadata = {
        "video_id": frame.get("video_id"),
        "frame_id": frame.get("frame_id"),
        "timestamp_ms": frame.get("timestamp_ms"),
        "scope": "landmark",
    }
    landmark_count = frame.get("landmark_count")
    if landmark_count != len(LANDMARK_NAMES):
        observations.append(
            {
                **metadata,
                "landmark_name": None,
                "field": None,
                "reason": f"landmark_count_mismatch:{landmark_count}",
                "value": landmark_count,
            }
        )
    if not frame.get("pose_detected", False):
        observations.append(
            {
                **metadata,
                "landmark_name": None,
                "field": None,
                "reason": "pose_not_detected",
                "value": None,
            }
        )
        return observations

    for landmark_name in LANDMARK_NAMES:
        for field in LANDMARK_FIELDS:
            value = frame.get(f"{landmark_name}_{field}")
            reason = None
            if value is None:
                reason = "field_unavailable" if field in {"visibility", "presence"} else "missing_coordinate"
            else:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = math.nan
                    reason = "not_numeric"
                if reason is None and not math.isfinite(numeric_value):
                    reason = "non_finite"
                if (
                    reason is None
                    and field in {"visibility", "presence"}
                    and not 0.0 <= numeric_value <= 1.0
                ):
                    reason = "score_out_of_range"
                elif reason is None and field == "visibility" and numeric_value < DEFAULT_VISIBILITY_THRESHOLD:
                    reason = "low_visibility"
                elif reason is None and field == "presence" and numeric_value < DEFAULT_PRESENCE_THRESHOLD:
                    reason = "low_presence"
            if reason is not None:
                observations.append(
                    {
                        **metadata,
                        "landmark_name": landmark_name,
                        "field": field,
                        "reason": reason,
                        "value": None if value is None else str(value),
                    }
                )
    return observations


def quality_frame(
    video_id: str | None,
    frame_id: str | None,
    timestamp_ms: int | None,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Add frame identifiers to a feature-level invalid observation."""
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "scope": observation.get("scope", "feature"),
        "landmark_name": observation.get("landmark_name"),
        "field": observation.get("field"),
        "feature_name": observation.get("feature_name"),
        "reason": observation.get("reason"),
        "value": observation.get("value"),
    }
