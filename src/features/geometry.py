"""Reusable 3D geometry operations for pose landmarks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Point3D = Sequence[float] | np.ndarray


def _as_point(point: Point3D | None) -> np.ndarray | None:
    if point is None:
        return None
    try:
        values = np.asarray(point, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != (3,) or not np.isfinite(values).all():
        return None
    return values


def center_point(points: Sequence[Point3D | None]) -> np.ndarray | None:
    """Return the centroid if every input is a finite 3D point."""
    if not points:
        return None
    converted = [_as_point(point) for point in points]
    if any(point is None for point in converted):
        return None
    return np.mean(np.stack(converted), axis=0)


def distance_3d(first: Point3D | None, second: Point3D | None) -> float | None:
    """Euclidean distance between two finite 3D points."""
    first_point = _as_point(first)
    second_point = _as_point(second)
    if first_point is None or second_point is None:
        return None
    return float(np.linalg.norm(first_point - second_point))


def angle_degrees(
    first: Point3D | None,
    vertex: Point3D | None,
    last: Point3D | None,
) -> float | None:
    """Angle at vertex, in degrees, using the dot product."""
    first_point = _as_point(first)
    vertex_point = _as_point(vertex)
    last_point = _as_point(last)
    if first_point is None or vertex_point is None or last_point is None:
        return None
    first_vector = first_point - vertex_point
    last_vector = last_point - vertex_point
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(last_vector)
    if denominator <= np.finfo(float).eps:
        return None
    cosine = np.dot(first_vector, last_vector) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def point_to_line_distance_3d(
    point: Point3D | None,
    line_start: Point3D | None,
    line_end: Point3D | None,
) -> float | None:
    """Shortest distance from a point to the infinite 3D line through two points."""
    point_value = _as_point(point)
    start_value = _as_point(line_start)
    end_value = _as_point(line_end)
    if point_value is None or start_value is None or end_value is None:
        return None
    direction = end_value - start_value
    length = np.linalg.norm(direction)
    if length <= np.finfo(float).eps:
        return None
    projection = start_value + direction * (
        np.dot(point_value - start_value, direction) / np.dot(direction, direction)
    )
    return float(np.linalg.norm(point_value - projection))


def point_to_line_vertical_deviation_3d(
    point: Point3D | None,
    line_start: Point3D | None,
    line_end: Point3D | None,
) -> float | None:
    """Absolute y-component of the point's residual from a 3D line."""
    point_value = _as_point(point)
    start_value = _as_point(line_start)
    end_value = _as_point(line_end)
    if point_value is None or start_value is None or end_value is None:
        return None
    direction = end_value - start_value
    squared_length = np.dot(direction, direction)
    if squared_length <= np.finfo(float).eps:
        return None
    projection = start_value + direction * (
        np.dot(point_value - start_value, direction) / squared_length
    )
    return float(abs(point_value[1] - projection[1]))
