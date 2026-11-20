"""Frame-level biomechanical features derived from world landmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.features.geometry import (
    angle_degrees,
    center_point,
    distance_3d,
    point_to_line_distance_3d,
    point_to_line_vertical_deviation_3d,
)

LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

PLANK_FEATURE_NAMES = (
    "shoulder_hip_ankle_angle",
    "shoulder_hip_knee_angle",
    "hip_knee_ankle_angle",
    "shoulder_elbow_wrist_angle",
    "elbow_shoulder_hip_angle",
    "head_neck_alignment",
    "hip_deviation_norm",
    "hip_vertical_deviation_norm",
)

HOLLOW_BODY_FEATURE_NAMES = (
    "shoulder_hip_knee_angle",
    "hip_knee_ankle_angle",
    "torso_orientation",
    "leg_elevation_angle",
    "arm_alignment_angle",
    "head_alignment_angle",
    "shoulder_hip_distance_norm",
    "hip_ankle_distance_norm",
)

ANGLE_FEATURE_NAMES = {
    "shoulder_hip_ankle_angle",
    "shoulder_hip_knee_angle",
    "hip_knee_ankle_angle",
    "shoulder_elbow_wrist_angle",
    "elbow_shoulder_hip_angle",
    "head_neck_alignment",
    "torso_orientation",
    "leg_elevation_angle",
    "arm_alignment_angle",
    "head_alignment_angle",
}

DEVIATION_FEATURE_NAMES = {"hip_deviation_norm", "hip_vertical_deviation_norm"}
DEFAULT_VISIBILITY_THRESHOLD = 0.5
DEFAULT_PRESENCE_THRESHOLD = 0.5


LANDMARK_GROUPS = {
    "shoulders": ("left_shoulder", "right_shoulder"),
    "hips": ("left_hip", "right_hip"),
    "knees": ("left_knee", "right_knee"),
    "ankles": ("left_ankle", "right_ankle"),
    "elbows": ("left_elbow", "right_elbow"),
    "wrists": ("left_wrist", "right_wrist"),
    "ears": ("left_ear", "right_ear"),
}


def _feature_groups(exercise: str) -> dict[str, tuple[str, ...]]:
    definitions = {
        "plank": {
            "shoulder_hip_ankle_angle": ("shoulders", "hips", "ankles"),
            "shoulder_hip_knee_angle": ("shoulders", "hips", "knees"),
            "hip_knee_ankle_angle": ("hips", "knees", "ankles"),
            "shoulder_elbow_wrist_angle": ("shoulders", "elbows", "wrists"),
            "elbow_shoulder_hip_angle": ("elbows", "shoulders", "hips"),
            "head_neck_alignment": ("ears", "shoulders", "hips"),
            "hip_deviation_norm": ("shoulders", "hips", "ankles"),
            "hip_vertical_deviation_norm": ("shoulders", "hips", "ankles"),
        },
        "hollow_body": {
            "shoulder_hip_knee_angle": ("shoulders", "hips", "knees"),
            "hip_knee_ankle_angle": ("hips", "knees", "ankles"),
            "torso_orientation": ("shoulders", "hips"),
            "leg_elevation_angle": ("hips", "ankles"),
            "arm_alignment_angle": ("wrists", "shoulders", "hips"),
            "head_alignment_angle": ("ears", "shoulders", "hips"),
            "shoulder_hip_distance_norm": ("shoulders", "hips", "ankles"),
            "hip_ankle_distance_norm": ("shoulders", "hips", "ankles"),
        },
    }
    if exercise not in definitions:
        raise ValueError(f"Ejercicio no soportado: {exercise}")
    return definitions[exercise]


def _feature_landmarks(exercise: str) -> dict[str, tuple[str, ...]]:
    definitions = _feature_groups(exercise)
    return {
        feature: tuple(
            landmark
            for group in groups
            for landmark in LANDMARK_GROUPS[group]
        )
        for feature, groups in definitions.items()
    }


def feature_landmark_requirements(exercise: str) -> dict[str, tuple[str, ...]]:
    """Return raw landmark names required to validate each feature."""
    return _feature_landmarks(exercise)


def _valid_point(
    landmarks: Mapping[str, Mapping[str, Any]],
    name: str,
    visibility_threshold: float,
    presence_threshold: float,
) -> tuple[np.ndarray | None, list[str]]:
    item = landmarks.get(name)
    if item is None:
        return None, [f"missing_landmark:{name}"]
    reasons = []
    coordinates = []
    for axis in ("x", "y", "z"):
        value = item.get(axis)
        if value is None:
            reasons.append(f"missing_coordinate:{name}.{axis}")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            reasons.append(f"invalid_coordinate:{name}.{axis}")
            continue
        if not math.isfinite(numeric_value):
            reasons.append(f"non_finite_coordinate:{name}.{axis}")
        coordinates.append(numeric_value)
    if len(coordinates) != 3 or reasons:
        return None, reasons
    for field, threshold in (
        ("visibility", visibility_threshold),
        ("presence", presence_threshold),
    ):
        score = item.get(field)
        # Some MediaPipe model versions do not provide these optional scores.
        if score is None:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            reasons.append(f"invalid_{field}:{name}")
            continue
        if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
            reasons.append(f"invalid_{field}:{name}")
        elif score_value < threshold:
            reasons.append(f"low_{field}:{name}")
    if reasons:
        return None, reasons
    return np.asarray(coordinates, dtype=float), []


def _mean_angle(
    landmarks: Mapping[str, np.ndarray],
    left: tuple[str, str, str],
    right: tuple[str, str, str],
) -> float | None:
    values = [
        angle_degrees(*(landmarks[name] for name in side))
        for side in (left, right)
        if all(name in landmarks for name in side)
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def _torso_length(points: Mapping[str, np.ndarray]) -> float | None:
    return distance_3d(points["shoulders"], points["hips"])


def _relative_angle_to_vertical(vector: np.ndarray) -> float | None:
    length = float(np.linalg.norm(vector))
    if length <= np.finfo(float).eps:
        return None
    cosine = abs(float(vector[1])) / length
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _leg_elevation(hip: np.ndarray, ankle: np.ndarray) -> float | None:
    vector = ankle - hip
    horizontal_length = float(np.hypot(vector[0], vector[2]))
    if horizontal_length <= np.finfo(float).eps and abs(float(vector[1])) <= np.finfo(float).eps:
        return None
    return float(np.degrees(np.arctan2(abs(float(vector[1])), horizontal_length)))


def extract_frame_features(
    frame: Mapping[str, Any],
    exercise: str,
    *,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    presence_threshold: float = DEFAULT_PRESENCE_THRESHOLD,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    """Calculate each feature independently and return invalid-feature observations."""
    feature_groups = _feature_groups(exercise)
    raw_landmarks = {
        name: {
            field: frame.get(f"{name}_{field}")
            for field in ("x", "y", "z", "visibility", "presence")
        }
        for name in LANDMARK_NAMES
    }

    points: dict[str, np.ndarray] = {}
    landmark_reasons: dict[str, list[str]] = {}
    for name in LANDMARK_NAMES:
        point, reasons = _valid_point(
            raw_landmarks,
            name,
            visibility_threshold,
            presence_threshold,
        )
        if point is not None:
            points[name] = point
        landmark_reasons[name] = reasons

    centers = {
        group: center_point(
            [points[name] for name in landmark_names if name in points]
        )
        for group, landmark_names in LANDMARK_GROUPS.items()
    }
    output: dict[str, float | None] = {}
    invalid: list[dict[str, Any]] = []

    def evaluate(
        feature: str,
        calculation: Any,
    ) -> None:
        missing_groups = [
            group for group in feature_groups[feature] if centers[group] is None
        ]
        reasons = [
            reason
            for group in missing_groups
            for name in LANDMARK_GROUPS[group]
            for reason in landmark_reasons[name]
        ]
        if reasons:
            output[feature] = None
            invalid.extend(
                {
                    "scope": "feature",
                    "feature_name": feature,
                    "reason": reason,
                }
                for reason in sorted(set(reasons))
            )
            return
        for group in feature_groups[feature]:
            unusable_sides = [
                name
                for name in LANDMARK_GROUPS[group]
                if landmark_reasons[name]
            ]
            if unusable_sides:
                invalid.append(
                    {
                        "scope": "feature",
                        "feature_name": feature,
                        "reason": f"partial_landmark_group:{group}",
                    }
                )
        value = calculation()
        if value is None or not math.isfinite(float(value)):
            output[feature] = None
            invalid.append(
                {
                    "scope": "feature",
                    "feature_name": feature,
                    "reason": "degenerate_geometry",
                }
            )
        else:
            output[feature] = float(value)

    def angle(left: str, vertex: str, right: str) -> float | None:
        return angle_degrees(centers[left], centers[vertex], centers[right])

    if exercise == "plank":
        evaluate("shoulder_hip_ankle_angle", lambda: angle("shoulders", "hips", "ankles"))
        evaluate("shoulder_hip_knee_angle", lambda: angle("shoulders", "hips", "knees"))
        evaluate("hip_knee_ankle_angle", lambda: angle("hips", "knees", "ankles"))
        evaluate(
            "shoulder_elbow_wrist_angle",
            lambda: _mean_angle(
                points,
                ("left_shoulder", "left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow", "right_wrist"),
            ),
        )
        evaluate(
            "elbow_shoulder_hip_angle",
            lambda: _mean_angle(
                points,
                ("left_elbow", "left_shoulder", "left_hip"),
                ("right_elbow", "right_shoulder", "right_hip"),
            ),
        )
        evaluate("head_neck_alignment", lambda: angle("ears", "shoulders", "hips"))

        def hip_deviation() -> float | None:
            torso = _torso_length(centers)
            distance = point_to_line_distance_3d(
                centers["hips"], centers["shoulders"], centers["ankles"]
            )
            return distance / torso if distance is not None and torso and torso > 0 else None

        def hip_vertical_deviation() -> float | None:
            torso = _torso_length(centers)
            distance = point_to_line_vertical_deviation_3d(
                centers["hips"], centers["shoulders"], centers["ankles"]
            )
            return distance / torso if distance is not None and torso and torso > 0 else None

        evaluate("hip_deviation_norm", hip_deviation)
        evaluate("hip_vertical_deviation_norm", hip_vertical_deviation)
    else:
        evaluate("shoulder_hip_knee_angle", lambda: angle("shoulders", "hips", "knees"))
        evaluate("hip_knee_ankle_angle", lambda: angle("hips", "knees", "ankles"))
        evaluate(
            "torso_orientation",
            lambda: _relative_angle_to_vertical(centers["shoulders"] - centers["hips"]),
        )
        evaluate(
            "leg_elevation_angle",
            lambda: _leg_elevation(centers["hips"], centers["ankles"]),
        )
        evaluate(
            "arm_alignment_angle",
            lambda: angle("wrists", "shoulders", "hips"),
        )
        evaluate(
            "head_alignment_angle",
            lambda: angle("ears", "shoulders", "hips"),
        )

        def shoulder_hip_distance_norm() -> float | None:
            shoulder_hip = distance_3d(centers["shoulders"], centers["hips"])
            shoulder_ankle = distance_3d(centers["shoulders"], centers["ankles"])
            return (
                shoulder_hip / shoulder_ankle
                if shoulder_hip is not None and shoulder_ankle and shoulder_ankle > 0
                else None
            )

        def hip_ankle_distance_norm() -> float | None:
            torso = _torso_length(centers)
            hip_ankle = distance_3d(centers["hips"], centers["ankles"])
            return hip_ankle / torso if hip_ankle is not None and torso and torso > 0 else None

        evaluate("shoulder_hip_distance_norm", shoulder_hip_distance_norm)
        evaluate("hip_ankle_distance_norm", hip_ankle_distance_norm)

    return output, invalid


FEATURE_DEFINITIONS = {
    "shoulder_hip_ankle_angle": {
        "definition": "Ángulo en el centro de caderas entre hombros y tobillos.",
        "formula": "angle(centro_hombros, centro_caderas, centro_tobillos)",
        "units": "degrees",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
    "shoulder_hip_knee_angle": {
        "definition": "Ángulo en el centro de caderas entre hombros y rodillas.",
        "formula": "angle(centro_hombros, centro_caderas, centro_rodillas)",
        "units": "degrees",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"],
    },
    "hip_knee_ankle_angle": {
        "definition": "Ángulo en el centro de rodillas entre caderas y tobillos.",
        "formula": "angle(centro_caderas, centro_rodillas, centro_tobillos)",
        "units": "degrees",
        "landmarks": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
    },
    "shoulder_elbow_wrist_angle": {
        "definition": "Media de los ángulos izquierdo y derecho en los codos.",
        "formula": "mean(angle(hombro, codo, muñeca) por lado)",
        "units": "degrees",
        "landmarks": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
    },
    "elbow_shoulder_hip_angle": {
        "definition": "Media de los ángulos izquierdo y derecho en los hombros.",
        "formula": "mean(angle(codo, hombro, cadera) por lado)",
        "units": "degrees",
        "landmarks": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    },
    "head_neck_alignment": {
        "definition": "Ángulo en los hombros entre el punto medio de orejas y caderas.",
        "formula": "angle(centro_orejas, centro_hombros, centro_caderas)",
        "units": "degrees",
        "landmarks": ["left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    },
    "hip_deviation_norm": {
        "definition": "Distancia 3D de caderas a la línea hombros-tobillos, dividida por longitud de torso.",
        "formula": "distancia_punto_linea(caderas, hombros, tobillos) / distancia(hombros, caderas)",
        "units": "torso_lengths",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
    "hip_vertical_deviation_norm": {
        "definition": "Componente vertical del residuo caderas-línea hombros-tobillos, normalizada por longitud de torso.",
        "formula": "abs(residuo_y(caderas, linea_hombros_tobillos)) / distancia(hombros, caderas)",
        "units": "torso_lengths",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
    "torso_orientation": {
        "definition": "Orientación del torso respecto al eje vertical del sistema de coordenadas world.",
        "formula": "acos(abs(vector_torso_y) / longitud(vector_torso))",
        "units": "degrees",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    },
    "leg_elevation_angle": {
        "definition": "Elevación del vector caderas-tobillos respecto al plano horizontal.",
        "formula": "atan2(abs(vector_y), norm(vector_xz))",
        "units": "degrees",
        "landmarks": ["left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
    "arm_alignment_angle": {
        "definition": "Ángulo en los hombros entre muñecas y caderas.",
        "formula": "angle(centro_munecas, centro_hombros, centro_caderas)",
        "units": "degrees",
        "landmarks": ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    },
    "head_alignment_angle": {
        "definition": "Ángulo en los hombros entre el punto medio de orejas y caderas.",
        "formula": "angle(centro_orejas, centro_hombros, centro_caderas)",
        "units": "degrees",
        "landmarks": ["left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    },
    "shoulder_hip_distance_norm": {
        "definition": "Distancia hombros-caderas normalizada por distancia hombros-tobillos.",
        "formula": "distancia(hombros, caderas) / distancia(hombros, tobillos)",
        "units": "dimensionless",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
    "hip_ankle_distance_norm": {
        "definition": "Distancia caderas-tobillos normalizada por longitud de torso hombros-caderas.",
        "formula": "distancia(caderas, tobillos) / distancia(hombros, caderas)",
        "units": "dimensionless",
        "landmarks": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"],
    },
}


def feature_dictionary(exercise: str) -> dict[str, Any]:
    """Build the documented feature dictionary for one supported exercise."""
    if exercise == "plank":
        names = PLANK_FEATURE_NAMES
    elif exercise == "hollow_body":
        names = HOLLOW_BODY_FEATURE_NAMES
    else:
        raise ValueError(f"Ejercicio no soportado: {exercise}")
    return {
        "exercise": exercise,
        "coordinate_system": "MediaPipe Pose World Landmarks; origen al centro de caderas; sin traslación adicional.",
        "features": {
            name: FEATURE_DEFINITIONS[name]
            for name in names
        },
    }
