"""Fixed-rate sliding feature windows and prediction stabilization."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.features.temporal import (
    MIN_VALID_FRAMES,
    TARGET_FPS,
    WINDOW_SECONDS,
    aggregate_feature_window,
)


class TemporalFeatureBuffer:
    """Keep the most recent 2-second feature samples at the configured sample rate."""

    def __init__(
        self,
        exercise: str,
        *,
        window_seconds: float = WINDOW_SECONDS,
        target_fps: float = TARGET_FPS,
        min_valid_frames: int = MIN_VALID_FRAMES,
    ) -> None:
        if exercise not in {"plank", "hollow_body"}:
            raise ValueError(f"Ejercicio no soportado: {exercise}")
        if window_seconds <= 0 or target_fps <= 0 or min_valid_frames < 1:
            raise ValueError("La ventana, frecuencia y mínimo de frames deben ser positivos.")
        self.exercise = exercise
        self.window_seconds = window_seconds
        self.target_fps = target_fps
        self.min_valid_frames = min_valid_frames
        self.capacity = int(round(window_seconds * target_fps))
        if self.capacity < min_valid_frames:
            raise ValueError("La ventana no puede contener el mínimo de frames requeridos.")
        self.samples: deque[dict[str, Any]] = deque(maxlen=self.capacity)

    @property
    def ready(self) -> bool:
        return len(self.samples) == self.capacity

    def clear(self) -> None:
        self.samples.clear()

    def append(
        self,
        timestamp_ms: int,
        features: Mapping[str, float | None],
    ) -> None:
        if self.samples and timestamp_ms <= self.samples[-1]["timestamp_ms"]:
            raise ValueError("Los timestamps de inferencia deben ser estrictamente crecientes.")
        self.samples.append(
            {
                "timestamp_ms": int(timestamp_ms),
                **dict(features),
            }
        )

    def model_input(
        self,
        feature_columns: Sequence[str],
    ) -> tuple[pd.DataFrame | None, list[str]]:
        """Return ordered model inputs only when every feature passes its frame minimum."""
        if not self.ready:
            return None, [f"insufficient_window_frames:{len(self.samples)}<{self.capacity}"]
        sample_frame = pd.DataFrame(list(self.samples))
        window_features, _ = aggregate_feature_window(
            sample_frame,
            self.exercise,
            window_start_ms=int(sample_frame["timestamp_ms"].iloc[0]),
            min_valid_frames=self.min_valid_frames,
        )
        insufficient = [
            column.removesuffix("_valid_frames")
            for column, value in window_features.items()
            if column.endswith("_valid_frames") and int(value or 0) < self.min_valid_frames
        ]
        if insufficient:
            return None, [f"insufficient_valid_feature:{name}" for name in insufficient]

        missing_columns = [column for column in feature_columns if column not in window_features]
        if missing_columns:
            raise ValueError(f"El modelo requiere características desconocidas: {missing_columns}")
        row = {column: window_features[column] for column in feature_columns}
        if any(value is None for value in row.values()):
            return None, ["invalid_temporal_feature"]
        return pd.DataFrame([row], columns=feature_columns), []


class PredictionStabilizer:
    """Majority-vote a bounded history of window predictions."""

    def __init__(self, history_size: int = 5) -> None:
        if history_size < 1:
            raise ValueError("history_size debe ser positivo.")
        self._predictions: deque[str] = deque(maxlen=history_size)

    def clear(self) -> None:
        self._predictions.clear()

    def add(self, prediction: str) -> str:
        self._predictions.append(prediction)
        counts = Counter(self._predictions)
        highest_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == highest_count}
        return next(
            label
            for label in reversed(self._predictions)
            if label in tied
        )
