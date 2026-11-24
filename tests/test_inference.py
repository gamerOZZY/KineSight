"""Tests for real-time window gating and bounded prediction smoothing."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.biomechanics import PLANK_FEATURE_NAMES
from src.inference.live_camera import run_camera_evaluation
from src.inference.temporal_buffer import PredictionStabilizer, TemporalFeatureBuffer


def model_feature_columns():
    return [
        f"{name}_{statistic}"
        for name in PLANK_FEATURE_NAMES
        for statistic in ("mean", "std", "min", "max", "range", "slope")
    ]


def valid_features(value=1.0):
    return {name: value for name in PLANK_FEATURE_NAMES}


class TemporalFeatureBufferTests(unittest.TestCase):
    def test_requires_full_window_and_returns_model_column_order(self):
        buffer = TemporalFeatureBuffer("plank")
        columns = model_feature_columns()
        self.assertIsNone(buffer.model_input(columns)[0])
        for index in range(10):
            buffer.append(index * 200, valid_features(index + 1.0))

        model_input, reasons = buffer.model_input(columns)

        self.assertTrue(buffer.ready)
        self.assertFalse(reasons)
        self.assertEqual(list(model_input.columns), columns)
        self.assertAlmostEqual(
            model_input["shoulder_hip_ankle_angle_mean"].iloc[0],
            5.5,
        )

    def test_insufficient_per_feature_samples_suppress_prediction(self):
        buffer = TemporalFeatureBuffer("plank")
        columns = model_feature_columns()
        for index in range(10):
            frame = valid_features()
            if index < 6:
                frame["hip_deviation_norm"] = None
            buffer.append(index * 200, frame)

        model_input, reasons = buffer.model_input(columns)

        self.assertIsNone(model_input)
        self.assertIn("insufficient_valid_feature:hip_deviation_norm", reasons)

    def test_timestamps_must_increase(self):
        buffer = TemporalFeatureBuffer("plank")
        buffer.append(100, valid_features())
        with self.assertRaises(ValueError):
            buffer.append(100, valid_features())


class PredictionStabilizerTests(unittest.TestCase):
    def test_majority_vote_uses_only_bounded_recent_predictions(self):
        stabilizer = PredictionStabilizer(history_size=3)
        self.assertEqual(stabilizer.add("correct"), "correct")
        self.assertEqual(stabilizer.add("incorrect"), "incorrect")
        self.assertEqual(stabilizer.add("incorrect"), "incorrect")
        self.assertEqual(stabilizer.add("correct"), "incorrect")
        self.assertEqual(stabilizer.add("correct"), "correct")

    def test_clear_discards_old_predictions(self):
        stabilizer = PredictionStabilizer(history_size=3)
        stabilizer.add("incorrect")
        stabilizer.clear()
        self.assertEqual(stabilizer.add("correct"), "correct")


class LiveCameraLifecycleTests(unittest.TestCase):
    def test_camera_is_released_when_application_quits(self):
        class FakeCapture:
            released = False

            def isOpened(self):
                return True

            def read(self):
                return True, np.zeros((64, 64, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        class FakeLandmarker:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def detect_for_video(self, *_):
                return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])

        capture = FakeCapture()
        with (
            patch("src.inference.live_camera.cv2.VideoCapture", return_value=capture),
            patch(
                "src.inference.live_camera.vision.PoseLandmarker.create_from_options",
                return_value=FakeLandmarker(),
            ),
            patch("src.inference.live_camera.cv2.imshow"),
            patch("src.inference.live_camera.cv2.waitKey", return_value=ord("q")),
            patch("src.inference.live_camera.cv2.destroyAllWindows"),
        ):
            run_camera_evaluation("plank", project_root=PROJECT_ROOT)

        self.assertTrue(capture.released)

    def test_unavailable_camera_is_reported(self):
        class FakeCapture:
            released = False

            def isOpened(self):
                return False

            def release(self):
                self.released = True

        capture = FakeCapture()
        with patch("src.inference.live_camera.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(OSError, "No se pudo abrir la cámara"):
                run_camera_evaluation("plank", project_root=PROJECT_ROOT)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
