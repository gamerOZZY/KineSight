"""MediaPipe Pose Landmarker VIDEO-mode extraction for Silver frames."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.features.biomechanics import LANDMARK_NAMES
from src.preprocessing.quality_control import (
    validate_landmark_frame,
    write_quality_parquet,
)

LOGGER = logging.getLogger(__name__)
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
FRAME_PATTERN = re.compile(r"^frame_(?P<number>\d+)\.jpg$", re.IGNORECASE)
VIDEO_NAME_PATTERN = re.compile(
    r"^(?P<quality>good|bad)_(?P<exercise>.+?)_(?P<number>\d+)_"
    r"(?:(?P<participant>p\d+)_)?(?P<camera_view>.+)$",
    re.IGNORECASE,
)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}
LANDMARK_ATTRIBUTES = ("x", "y", "z", "visibility", "presence")
LANDMARK_COLUMNS = tuple(
    f"{landmark}_{attribute}"
    for landmark in LANDMARK_NAMES
    for attribute in LANDMARK_ATTRIBUTES
)
SILVER_METADATA_COLUMNS = (
    "video_id",
    "participant_id",
    "exercise",
    "frame_id",
    "source_frame_index",
    "timestamp_ms",
    "label",
    "error_type",
    "camera_view",
    "source_video",
    "pose_detected",
    "landmark_count",
    "processing_status",
)
SILVER_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("participant_id", pa.string()),
        ("exercise", pa.string()),
        ("frame_id", pa.string()),
        ("source_frame_index", pa.int64()),
        ("timestamp_ms", pa.int64()),
        ("label", pa.string()),
        ("error_type", pa.string()),
        ("camera_view", pa.string()),
        ("source_video", pa.string()),
        ("pose_detected", pa.bool_()),
        ("landmark_count", pa.int64()),
        ("processing_status", pa.string()),
        *[(column, pa.float64()) for column in LANDMARK_COLUMNS],
    ]
)


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    participant_id: str | None
    exercise: str
    label: str
    error_type: str | None
    camera_view: str
    project_root: Path
    source_video: Path
    frame_directory: Path


def _metadata_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def discover_video_records(
    project_root: Path,
    exercise: str,
) -> list[VideoRecord]:
    """Pair extracted-frame directories with their Bronze videos and metadata."""
    exercise = exercise.lower()
    video_root = project_root / "data" / "bronze" / "videos" / exercise
    frame_root = project_root / "data" / "silver" / "extracted_frames" / exercise
    metadata_path = project_root / "data" / "bronze" / "metadata" / f"{exercise}_metadata.csv"
    if not video_root.is_dir() or not frame_root.is_dir():
        return []
    video_paths = []
    for label in ("correct", "incorrect"):
        label_root = video_root / label
        if label_root.is_dir():
            video_paths.extend(
                path
                for path in label_root.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
    video_paths.sort(key=lambda path: (path.parent.name.lower(), path.name.lower()))
    if not video_paths:
        return []
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Hay videos de {exercise}, pero no existe su metadata: {metadata_path}"
        )
    metadata = pd.read_csv(metadata_path)
    if len(video_paths) != len(metadata):
        raise ValueError(
            f"{exercise}: hay {len(video_paths)} videos y {len(metadata)} filas "
            f"en {metadata_path}; regenera o corrige la metadata."
        )

    exercise_column = "exercise" if "exercise" in metadata else "excercise"
    records = []
    for video_path, row in zip(video_paths, metadata.itertuples(index=False)):
        match = VIDEO_NAME_PATTERN.fullmatch(video_path.stem)
        if match is None:
            raise ValueError(f"Nombre de video no reconocido: {video_path.name}")
        video_id = _metadata_value(getattr(row, "video_id", None))
        if video_id is None or not re.fullmatch(r"P\d{2,}", video_id):
            raise ValueError(f"video_id ausente o inválido para {video_path.name}")
        frame_directory = frame_root / video_id
        if not frame_directory.is_dir():
            raise FileNotFoundError(f"No hay frames extraídos para {video_id}: {frame_directory}")

        filename_exercise = match.group("exercise").lower()
        filename_exercise = {"hollow": "hollow_body"}.get(
            filename_exercise,
            filename_exercise,
        )
        filename_label = {"good": "correct", "bad": "incorrect"}[
            match.group("quality").lower()
        ]
        metadata_label = _metadata_value(getattr(row, "label", None))
        metadata_exercise = _metadata_value(getattr(row, exercise_column, None))
        if metadata_label != video_path.parent.name.lower() or metadata_label != filename_label:
            raise ValueError(f"Etiqueta metadata/video inconsistente: {video_path}")
        if metadata_exercise is None or metadata_exercise.lower() != exercise:
            raise ValueError(f"Ejercicio metadata/video inconsistente: {video_path}")
        if filename_exercise != exercise:
            raise ValueError(f"Nombre de video de ejercicio inesperado: {video_path.name}")

        participant_id = _metadata_value(
            getattr(row, "participant_id", getattr(row, "Participant_id", None))
        )
        filename_participant = match.group("participant")
        filename_participant = filename_participant.upper() if filename_participant else None
        if participant_id != filename_participant:
            raise ValueError(f"Participant_id metadata/video inconsistente: {video_path}")
        camera_view = _metadata_value(getattr(row, "camera_view", None))
        if camera_view is None:
            raise ValueError(f"camera_view ausente para {video_path.name}")
        if camera_view.lower() != match.group("camera_view").lower():
            raise ValueError(f"camera_view metadata/video inconsistente: {video_path}")
        records.append(
            VideoRecord(
                video_id=video_id,
                participant_id=participant_id,
                exercise=exercise,
                label=metadata_label,
                error_type=_metadata_value(getattr(row, "error_type", None)),
                camera_view=camera_view,
                project_root=project_root,
                source_video=video_path,
                frame_directory=frame_directory,
            )
        )
    return records


def _ordered_frames(frame_directory: Path) -> list[tuple[int, Path]]:
    frames = []
    for path in frame_directory.glob("frame_*.jpg"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if match:
            frames.append((int(match.group("number")), path))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise FileNotFoundError(f"No se encontraron JPEG de frames en {frame_directory}")
    numbers = [number for number, _ in frames]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"Los IDs de frame no son consecutivos en {frame_directory}")
    return frames


def _source_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"OpenCV no pudo abrir el video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"FPS no válido para {video_path}: {fps}")
    return fps


def source_frame_index(sample_index: int, source_fps: float, sampled_fps: float) -> int:
    """Reconstruct the original CFR frame selected by the Silver frame sampler."""
    if sample_index < 0 or source_fps <= 0 or sampled_fps <= 0:
        raise ValueError("sample_index y frecuencias deben ser valores válidos.")
    if source_fps <= sampled_fps:
        return sample_index
    return int(math.ceil(sample_index * source_fps / sampled_fps - 1e-9))


def _safe_landmark_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def _new_row(record: VideoRecord, frame_id: str, source_index: int, timestamp_ms: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "video_id": record.video_id,
        "participant_id": record.participant_id,
        "exercise": record.exercise,
        "frame_id": frame_id,
        "source_frame_index": source_index,
        "timestamp_ms": timestamp_ms,
        "label": record.label,
        "error_type": record.error_type,
        "camera_view": record.camera_view,
        "source_video": record.source_video.relative_to(record.project_root).as_posix(),
        "pose_detected": False,
        "landmark_count": 0,
        "processing_status": "no_pose",
    }
    row.update({column: None for column in LANDMARK_COLUMNS})
    return row


def _landmark_observations(
    world_landmarks: list[Any],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    row["pose_detected"] = bool(world_landmarks)
    row["landmark_count"] = len(world_landmarks)
    row["processing_status"] = (
        "ok" if len(world_landmarks) == len(LANDMARK_NAMES)
        else "partial_pose" if world_landmarks
        else "no_pose"
    )
    raw_invalid = []
    for index, landmark_name in enumerate(LANDMARK_NAMES):
        landmark = world_landmarks[index] if index < len(world_landmarks) else None
        for attribute in LANDMARK_ATTRIBUTES:
            value = getattr(landmark, attribute, None) if landmark is not None else None
            safe_value = _safe_landmark_value(value)
            row[f"{landmark_name}_{attribute}"] = safe_value
            if value is not None and safe_value is None:
                raw_invalid.append(
                    {
                        "video_id": row["video_id"],
                        "frame_id": row["frame_id"],
                        "timestamp_ms": row["timestamp_ms"],
                        "scope": "landmark",
                        "landmark_name": landmark_name,
                        "field": attribute,
                        "feature_name": None,
                        "reason": "non_finite_or_non_numeric_raw_value",
                        "value": str(value),
                    }
                )
    return raw_invalid


def _make_landmarker(model_path: Path) -> Any:
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_video_landmarks(
    record: VideoRecord,
    model_path: Path,
    *,
    sampled_fps: float = 7.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Extract one Silver row per input frame, preserving failed detections."""
    if sampled_fps <= 0 or not math.isfinite(sampled_fps):
        raise ValueError(f"sampled_fps debe ser positivo y finito: {sampled_fps}")
    fps = _source_fps(record.source_video)
    frames = _ordered_frames(record.frame_directory)
    rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    previous_timestamp = -1

    with _make_landmarker(model_path) as landmarker:
        for sample_index, (frame_number, frame_path) in enumerate(frames):
            source_index = source_frame_index(sample_index, fps, sampled_fps)
            timestamp_ms = int(round(source_index * 1000.0 / fps))
            if timestamp_ms <= previous_timestamp:
                raise ValueError(
                    f"Timestamps no monótonos en {record.video_id}/{frame_path.name}: "
                    f"{timestamp_ms} <= {previous_timestamp}"
                )
            previous_timestamp = timestamp_ms
            row = _new_row(
                record,
                frame_path.stem,
                source_index,
                timestamp_ms,
            )
            image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                row["processing_status"] = "error"
                quality.append(
                    {
                        "video_id": record.video_id,
                        "frame_id": frame_path.stem,
                        "timestamp_ms": timestamp_ms,
                        "scope": "frame",
                        "landmark_name": None,
                        "field": None,
                        "feature_name": None,
                        "reason": "image_decode_failed",
                        "value": str(frame_path),
                    }
                )
                LOGGER.error("No se pudo decodificar el frame %s", frame_path)
                rows.append(row)
                continue
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                poses = result.pose_world_landmarks or []
                raw_invalid = _landmark_observations(poses[0] if poses else [], row)
                quality.extend(raw_invalid)
            except Exception as error:
                # Preserve a row for every frame and make the failed observation auditable.
                LOGGER.exception("Fallo de MediaPipe en %s", frame_path)
                row["processing_status"] = "error"
                quality.append(
                    {
                        "video_id": record.video_id,
                        "frame_id": frame_path.stem,
                        "timestamp_ms": timestamp_ms,
                        "scope": "frame",
                        "landmark_name": None,
                        "field": None,
                        "feature_name": None,
                        "reason": "pose_landmarker_error",
                        "value": f"{type(error).__name__}: {error}",
                    }
                )
            quality.extend(validate_landmark_frame(row))
            rows.append(row)
    columns = [*SILVER_METADATA_COLUMNS, *LANDMARK_COLUMNS]
    return pd.DataFrame(rows, columns=columns), quality


def extract_exercise_landmarks(
    project_root: Path,
    exercise: str,
    model_path: Path,
    *,
    sampled_fps: float = 7.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one Parquet file per video plus a consolidated Silver QC file."""
    exercise = exercise.lower()
    records = discover_video_records(project_root, exercise)
    output_root = project_root / "data" / "silver" / "landmarks" / exercise
    quality_path = output_root / "quality_observations.parquet"
    outputs = [output_root / f"{record.video_id}.parquet" for record in records]
    if not records:
        LOGGER.info("No hay videos/frames para %s; no se generan landmarks.", exercise)
        return {"exercise": exercise, "videos": 0, "frames": 0, "quality_observations": 0}
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo Pose Landmarker: {model_path}. "
            f"Descarga el modelo Lite oficial desde {MODEL_DOWNLOAD_URL}."
        )
    existing = [path for path in [*outputs, quality_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Ya existen resultados Silver; usa overwrite=True solo si deseas reemplazarlos: "
            + ", ".join(str(path) for path in existing)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    all_quality: list[dict[str, Any]] = []
    total_frames = 0
    for record, output_path in zip(records, outputs):
        landmarks, observations = extract_video_landmarks(
            record,
            model_path,
            sampled_fps=sampled_fps,
        )
        table = pa.Table.from_pandas(landmarks, schema=SILVER_SCHEMA, preserve_index=False)
        pq.write_table(table, output_path)
        total_frames += len(landmarks)
        all_quality.extend(observations)
        LOGGER.info(
            "Landmarks de %s: %d filas escritas en %s",
            record.video_id,
            len(landmarks),
            output_path,
        )
    write_quality_parquet(quality_path, all_quality)
    LOGGER.info(
        "QC de %s: %d observaciones inválidas registradas en %s",
        exercise,
        len(all_quality),
        quality_path,
    )
    return {
        "exercise": exercise,
        "videos": len(records),
        "frames": total_frames,
        "quality_observations": len(all_quality),
        "output_directory": output_root,
    }
