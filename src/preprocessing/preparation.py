"""Reproducible Bronze metadata creation and Silver frame extraction."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}
VIDEO_NAME_PATTERN = re.compile(
    r"^(?P<quality>good|bad)_(?P<exercise>[^_]+)_(?P<number>\d+)_"
    r"(?:(?P<participant>p\d+)_)?(?P<camera_view>.+)$",
    re.IGNORECASE,
)
EXERCISE_FILENAME_ALIASES = {"plank": "plank", "hollow": "hollow_body"}
METADATA_COLUMNS = (
    "video_id",
    "Participant_id",
    "excercise",
    "label",
    "camera_view",
    "duration_in_seconds",
    "fps",
)
LABEL_BY_QUALITY = {"good": "correct", "bad": "incorrect"}
QUALITY_BY_FOLDER = {"correct": "good", "incorrect": "bad"}


def _video_paths(video_root: Path) -> list[Path]:
    if not video_root.is_dir():
        return []
    paths = [
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: (path.parent.name.lower(), path.name.lower()))


def _parse_video(path: Path, exercise: str) -> dict[str, Any]:
    match = VIDEO_NAME_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"El nombre del video no sigue el formato esperado: {path.name}")
    expected_folder = path.parent.name.lower()
    if expected_folder not in QUALITY_BY_FOLDER:
        raise ValueError(f"El video debe estar en correct/incorrect: {path}")
    label = LABEL_BY_QUALITY[match.group("quality").lower()]
    if label != expected_folder:
        raise ValueError(f"Etiqueta del nombre y carpeta no coinciden: {path}")
    filename_exercise = EXERCISE_FILENAME_ALIASES.get(
        match.group("exercise").lower(),
        match.group("exercise").lower(),
    )
    if filename_exercise != exercise:
        raise ValueError(
            f"El video {path.name} es de {filename_exercise}, no de {exercise}."
        )
    return {
        "path": path,
        "Participant_id": (
            match.group("participant").upper() if match.group("participant") else ""
        ),
        "excercise": exercise,
        "label": label,
        "camera_view": match.group("camera_view"),
    }


def _video_properties(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"OpenCV no pudo abrir el video: {path}")
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if (
        not math.isfinite(frame_count)
        or not math.isfinite(fps)
        or frame_count <= 0
        or fps <= 0
    ):
        raise ValueError(
            f"Duración/FPS inválidos para {path.name}: cuadros={frame_count}, FPS={fps}"
        )
    return int(round(frame_count)), float(fps)


def create_exercise_metadata(
    project_root: Path,
    exercise: str,
    *,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Create an exercise CSV once, or validate an existing CSV without replacing it."""
    exercise = exercise.lower()
    video_root = project_root / "data" / "bronze" / "videos" / exercise
    metadata_path = project_root / "data" / "bronze" / "metadata" / f"{exercise}_metadata.csv"
    videos = [_parse_video(path, exercise) for path in _video_paths(video_root)]
    if not videos:
        LOGGER.info("No se encontraron videos de %s en Bronze.", exercise)
        return pd.DataFrame(columns=METADATA_COLUMNS), metadata_path

    rows = []
    for index, video in enumerate(videos, start=1):
        frame_count, fps = _video_properties(video["path"])
        rows.append(
            {
                "video_id": f"P{index:02d}",
                "Participant_id": video["Participant_id"],
                "excercise": video["excercise"],
                "label": video["label"],
                "camera_view": video["camera_view"],
                "duration_in_seconds": round(frame_count / fps, 3),
                "fps": round(fps, 3),
            }
        )
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)

    if metadata_path.exists() and not overwrite:
        existing = pd.read_csv(metadata_path)
        if list(existing.columns) != list(METADATA_COLUMNS) or len(existing) != len(metadata):
            raise FileExistsError(
                f"La metadata existente no coincide con los videos y no se reemplazará: "
                f"{metadata_path}"
            )
        for column in METADATA_COLUMNS:
            if column in {"fps", "duration_in_seconds"}:
                matches = (
                    pd.to_numeric(existing[column], errors="coerce").round(3)
                    == pd.to_numeric(metadata[column], errors="coerce").round(3)
                ).all()
            else:
                left = existing[column].fillna("").astype(str).str.strip()
                right = metadata[column].fillna("").astype(str).str.strip()
                matches = left.equals(right)
            if not matches:
                raise FileExistsError(
                    f"La columna {column} de la metadata no coincide con los videos; "
                    f"no se reemplazará {metadata_path}"
                )
        LOGGER.info(
            "Metadata existente validada: %s (%d videos).",
            metadata_path,
            len(metadata),
        )
        return metadata, metadata_path

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")
    LOGGER.info("Metadata creada: %s (%d videos).", metadata_path, len(metadata))
    return metadata, metadata_path


def _expected_sample_count(frame_count: int, fps: float, target_fps: float) -> int:
    interval = max(fps / target_fps, 1.0)
    return min(frame_count, math.floor((frame_count - 1) / interval) + 1)


def extract_exercise_frames(
    project_root: Path,
    exercise: str,
    *,
    target_fps: float = 7.0,
) -> dict[str, int]:
    """Save 7-FPS JPEG samples without deleting or replacing existing frames."""
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"target_fps debe ser positivo y finito: {target_fps}")
    exercise = exercise.lower()
    video_root = project_root / "data" / "bronze" / "videos" / exercise
    frame_root = project_root / "data" / "silver" / "extracted_frames" / exercise
    metadata_path = project_root / "data" / "bronze" / "metadata" / f"{exercise}_metadata.csv"
    videos = [_parse_video(path, exercise) for path in _video_paths(video_root)]
    if not videos:
        return {"videos": 0, "frames": 0, "created_videos": 0, "skipped_videos": 0}
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Primero crea la metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(videos):
        raise ValueError(
            f"Hay {len(videos)} videos y {len(metadata)} filas de metadata para {exercise}."
        )

    frame_root.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    created_videos = 0
    skipped_videos = 0
    for video, row in zip(videos, metadata.itertuples(index=False)):
        if str(row.excercise).lower() != exercise:
            raise ValueError(f"Ejercicio en metadata no coincide con {video['path'].name}")
        if str(row.label).lower() != video["label"]:
            raise ValueError(f"Etiqueta en metadata no coincide con {video['path'].name}")
        if str(row.camera_view).lower() != video["camera_view"].lower():
            raise ValueError(f"Vista de cámara en metadata no coincide con {video['path'].name}")
        expected_participant = video["Participant_id"]
        actual_participant = "" if pd.isna(row.Participant_id) else str(row.Participant_id)
        if actual_participant != expected_participant:
            raise ValueError(f"Participante en metadata no coincide con {video['path'].name}")

        video_id = str(row.video_id)
        output_dir = frame_root / video_id
        frame_count, fps = _video_properties(video["path"])
        expected_frames = _expected_sample_count(frame_count, fps, target_fps)
        existing_frames = sorted(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []
        if existing_frames:
            if len(existing_frames) == expected_frames and [
                path.name for path in existing_frames
            ] == [f"frame_{index:06d}.jpg" for index in range(1, expected_frames + 1)]:
                LOGGER.info("Frames de %s ya completos; se conservan sin cambios.", video_id)
                total_frames += expected_frames
                skipped_videos += 1
                continue
            raise FileExistsError(
                f"Hay una extracción parcial o con otra frecuencia en {output_dir}; "
                "no se borrará ni sobrescribirá."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video["path"]))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"OpenCV no pudo abrir el video: {video['path']}")
        saved_frames = 0
        source_frame_index = 0
        next_sample_frame = 0.0
        source_frame_interval = fps / target_fps
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if source_frame_index + 1e-9 >= next_sample_frame:
                    saved_frames += 1
                    output_path = output_dir / f"frame_{saved_frames:06d}.jpg"
                    if not cv2.imwrite(str(output_path), frame):
                        raise OSError(f"No se pudo escribir el frame {output_path}")
                    next_sample_frame += source_frame_interval
                source_frame_index += 1
        finally:
            capture.release()
        if saved_frames != expected_frames:
            raise RuntimeError(
                f"Extracción incompleta en {video_id}: {saved_frames} frames; "
                f"se esperaban {expected_frames}."
            )
        total_frames += saved_frames
        created_videos += 1
        LOGGER.info(
            "Frames extraídos para %s: %d a %.2f FPS.",
            video_id,
            saved_frames,
            target_fps,
        )
    return {
        "videos": len(videos),
        "frames": total_frames,
        "created_videos": created_videos,
        "skipped_videos": skipped_videos,
    }


def prepare_exercise(
    project_root: Path,
    exercise: str,
    *,
    target_fps: float = 7.0,
) -> dict[str, Any]:
    """Prepare metadata and extracted Silver frames from immutable Bronze videos."""
    metadata, metadata_path = create_exercise_metadata(project_root, exercise)
    frames = extract_exercise_frames(project_root, exercise, target_fps=target_fps)
    return {
        "exercise": exercise,
        "metadata_path": metadata_path,
        "metadata_rows": len(metadata),
        **frames,
    }
