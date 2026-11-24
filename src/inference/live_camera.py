"""OpenCV camera application using the trained Gold-compatible feature pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.features.biomechanics import LANDMARK_NAMES, extract_frame_features
from src.inference.temporal_buffer import PredictionStabilizer, TemporalFeatureBuffer

LOGGER = logging.getLogger(__name__)
MODEL_DIRECTORY = {
    "plank": "plank",
    "hollow_body": "hollow_body",
}
LABEL_TEXT = {"correct": "CORRECT", "incorrect": "INCORRECT"}
QUALITY_MESSAGE = "No se puede evaluar: mejora la visibilidad de tu cuerpo."
INCORRECT_MESSAGE = "Postura incorrecta. Revisa tu alineación corporal."


def _read_model(project_root: Path, exercise: str) -> tuple[Any, dict[str, Any]]:
    model_directory = project_root / "models" / MODEL_DIRECTORY[exercise]
    model_path = model_directory / "model.joblib"
    metadata_path = model_directory / "model_metadata.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo entrenado: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No existe la metadata del modelo: {metadata_path}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("exercise") != exercise:
        raise ValueError(f"El artefacto de modelo no corresponde a {exercise}.")
    feature_columns = metadata.get("features")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("La metadata no contiene el orden de características entrenado.")
    fitted_columns = list(getattr(model, "feature_names_in_", []))
    if fitted_columns and fitted_columns != feature_columns:
        raise ValueError("El orden de características del Pipeline no coincide con metadata.")
    return model, metadata


def _pose_options(model_path: Path) -> vision.PoseLandmarkerOptions:
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo Pose Landmarker: {model_path}")
    return vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _frame_features(result: Any, exercise: str) -> tuple[dict[str, float | None], bool]:
    frame: dict[str, Any] = {"pose_detected": False, "landmark_count": 0}
    for landmark_name in LANDMARK_NAMES:
        for attribute in ("x", "y", "z", "visibility", "presence"):
            frame[f"{landmark_name}_{attribute}"] = None

    poses = result.pose_world_landmarks or []
    if not poses:
        return {name: None for name in _feature_names(exercise)}, False

    world_landmarks = poses[0]
    frame["pose_detected"] = True
    frame["landmark_count"] = len(world_landmarks)
    for index, landmark_name in enumerate(LANDMARK_NAMES):
        if index >= len(world_landmarks):
            continue
        landmark = world_landmarks[index]
        for attribute in ("x", "y", "z", "visibility", "presence"):
            value = getattr(landmark, attribute, None)
            if value is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None and np.isfinite(numeric):
                    frame[f"{landmark_name}_{attribute}"] = numeric
    features, _ = extract_frame_features(frame, exercise)
    return features, len(world_landmarks) == len(LANDMARK_NAMES)


def _feature_names(exercise: str) -> tuple[str, ...]:
    from src.features.biomechanics import HOLLOW_BODY_FEATURE_NAMES, PLANK_FEATURE_NAMES

    return PLANK_FEATURE_NAMES if exercise == "plank" else HOLLOW_BODY_FEATURE_NAMES


def _draw_skeleton(image: np.ndarray, normalized_landmarks: list[Any] | None) -> None:
    if not normalized_landmarks:
        return
    height, width = image.shape[:2]
    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
    for connection in connections:
        start = normalized_landmarks[int(connection.start)]
        end = normalized_landmarks[int(connection.end)]
        start_xy = (int(start.x * width), int(start.y * height))
        end_xy = (int(end.x * width), int(end.y * height))
        cv2.line(image, start_xy, end_xy, (70, 220, 100), 2, cv2.LINE_AA)
    for landmark in normalized_landmarks:
        if not np.isfinite([landmark.x, landmark.y]).all():
            continue
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(image, point, 3, (40, 100, 255), -1, cv2.LINE_AA)


def _draw_status(
    image: np.ndarray,
    exercise: str,
    pose_detected: bool,
    status_text: str,
    provisional: bool,
) -> None:
    title = "PLANK" if exercise == "plank" else "HOLLOW BODY HOLD"
    cv2.rectangle(image, (8, 8), (image.shape[1] - 8, 130), (20, 20, 20), -1)
    cv2.putText(image, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(
        image,
        f"Pose: {'DETECTADA' if pose_detected else 'NO DETECTADA'}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (90, 220, 110) if pose_detected else (60, 120, 255),
        2,
    )
    cv2.putText(
        image,
        status_text[:90],
        (20, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    if provisional:
        cv2.putText(
            image,
            "MODELO EXPERIMENTAL - validar con mas participantes",
            (20, 123),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 190, 255),
            1,
        )


def run_camera_evaluation(
    exercise: str,
    *,
    project_root: Path | None = None,
    camera_index: int = 0,
) -> None:
    """Run one exercise model and its own sliding 2-second feature window."""
    if exercise not in MODEL_DIRECTORY:
        raise ValueError(f"Ejercicio no soportado: {exercise}")
    project_root = project_root or Path(__file__).resolve().parents[2]
    model, model_metadata = _read_model(project_root, exercise)
    feature_columns = model_metadata["features"]
    pose_model_path = project_root / "models" / "pose" / "pose_landmarker.task"
    options = _pose_options(pose_model_path)

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise OSError(
            f"No se pudo abrir la cámara {camera_index}. "
            "Comprueba que esté conectada y que ninguna otra aplicación la use."
        )

    feature_buffer = TemporalFeatureBuffer(exercise)
    stabilizer = PredictionStabilizer(history_size=5)
    last_status = QUALITY_MESSAGE
    last_prediction: str | None = None
    next_sample_ms: int | None = None
    previous_timestamp_ms = -1
    session_start = time.monotonic()
    window_name = f"KineSight - {exercise}"
    LOGGER.info(
        "Iniciando cámara para %s con modelo %s (provisional=%s)",
        exercise,
        model_metadata.get("algorithm"),
        model_metadata.get("provisional_selection"),
    )
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, image_bgr = capture.read()
                if not success:
                    LOGGER.error("No se pudo leer un frame de la cámara.")
                    break
                elapsed_ms = int(round((time.monotonic() - session_start) * 1000))
                timestamp_ms = max(elapsed_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                poses = result.pose_landmarks or []
                _draw_skeleton(image_bgr, poses[0] if poses else None)

                frame_features, full_pose = _frame_features(result, exercise)
                if next_sample_ms is None:
                    next_sample_ms = timestamp_ms
                new_sample = timestamp_ms >= next_sample_ms
                if new_sample:
                    feature_buffer.append(timestamp_ms, frame_features)
                    next_sample_ms = timestamp_ms + int(round(1000 / feature_buffer.target_fps))

                pose_detected = bool(poses) and full_pose
                if not pose_detected:
                    last_status = QUALITY_MESSAGE
                    last_prediction = None
                    stabilizer.clear()
                elif new_sample:
                    model_input, _ = feature_buffer.model_input(feature_columns)
                    if model_input is None:
                        last_status = QUALITY_MESSAGE
                        last_prediction = None
                        stabilizer.clear()
                    else:
                        raw_prediction = str(model.predict(model_input)[0])
                        last_prediction = stabilizer.add(raw_prediction)
                        last_status = (
                            "Evaluación: CORRECT"
                            if last_prediction == "correct"
                            else INCORRECT_MESSAGE
                        )

                _draw_status(
                    image_bgr,
                    exercise,
                    pose_detected,
                    last_status,
                    bool(model_metadata.get("provisional_selection")),
                )
                cv2.putText(
                    image_bgr,
                    "Presiona Q para salir",
                    (16, image_bgr.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow(window_name, image_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(exercise: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=f"Evaluación experimental en vivo: {exercise}.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara OpenCV.")
    args = parser.parse_args()
    run_camera_evaluation(exercise, camera_index=args.camera)
