"""Tests for frame-level biomechanical validity and temporal aggregation."""

import unittest

import pandas as pd

from src.features.biomechanics import (
    HOLLOW_BODY_FEATURE_NAMES,
    LANDMARK_NAMES,
    PLANK_FEATURE_NAMES,
    extract_frame_features,
)
from src.features.temporal import build_temporal_windows


def make_frame(**overrides):
    coordinates = {
        "left_shoulder": (-0.2, 0.0, 0.0),
        "right_shoulder": (0.2, 0.0, 0.0),
        "left_hip": (-0.15, 1.0, 0.0),
        "right_hip": (0.15, 1.0, 0.0),
        "left_knee": (-0.1, 1.5, 0.0),
        "right_knee": (0.1, 1.5, 0.0),
        "left_ankle": (-0.1, 2.0, 0.0),
        "right_ankle": (0.1, 2.0, 0.0),
        "left_elbow": (-0.4, 0.1, 0.0),
        "right_elbow": (0.4, 0.1, 0.0),
        "left_wrist": (-0.5, 0.3, 0.0),
        "right_wrist": (0.5, 0.3, 0.0),
        "left_ear": (-0.05, -0.2, 0.0),
        "right_ear": (0.05, -0.2, 0.0),
    }
    frame = {
        "video_id": "P01",
        "participant_id": "P1",
        "exercise": "plank",
        "camera_view": "side",
        "source_video": "data/bronze/videos/plank/correct/video.mp4",
        "label": "correct",
        "error_type": None,
        "frame_id": "frame_000001",
        "timestamp_ms": 0,
        "pose_detected": True,
        "landmark_count": 33,
    }
    for index, name in enumerate(LANDMARK_NAMES):
        x, y, z = coordinates.get(name, (index / 100, index / 100, 0.05))
        frame.update(
            {
                f"{name}_x": x,
                f"{name}_y": y,
                f"{name}_z": z,
                f"{name}_visibility": 0.9,
                f"{name}_presence": 0.95,
            }
        )
    frame.update(overrides)
    return frame


class BiomechanicalFeatureTests(unittest.TestCase):
    def test_feature_sets_match_exercise_contract(self):
        frame = make_frame()
        plank, plank_invalid = extract_frame_features(frame, "plank")
        hollow, hollow_invalid = extract_frame_features(frame, "hollow_body")
        self.assertEqual(tuple(plank), PLANK_FEATURE_NAMES)
        self.assertEqual(tuple(hollow), HOLLOW_BODY_FEATURE_NAMES)
        self.assertFalse(plank_invalid)
        self.assertFalse(hollow_invalid)
        self.assertTrue(all(value is not None for value in plank.values()))
        self.assertTrue(all(value is not None for value in hollow.values()))

    def test_single_visible_side_keeps_feature_and_records_partial_quality(self):
        frame = make_frame(right_ankle_visibility=0.1)
        features, invalid = extract_frame_features(frame, "plank")
        self.assertIsNotNone(features["shoulder_hip_ankle_angle"])
        self.assertTrue(
            any(
                observation["reason"] == "partial_landmark_group:ankles"
                for observation in invalid
            )
        )

    def test_missing_ankles_invalidates_only_dependent_features(self):
        frame = make_frame(
            left_ankle_x=None,
            left_ankle_y=None,
            left_ankle_z=None,
            right_ankle_x=None,
            right_ankle_y=None,
            right_ankle_z=None,
        )
        features, _ = extract_frame_features(frame, "plank")
        self.assertIsNone(features["shoulder_hip_ankle_angle"])
        self.assertIsNotNone(features["head_neck_alignment"])


class TemporalWindowTests(unittest.TestCase):
    def test_windows_do_not_mix_labels(self):
        rows = []
        for label in ("correct", "incorrect"):
            for frame_index in range(10):
                frame = make_frame()
                features, _ = extract_frame_features(frame, "plank")
                rows.append(
                    {
                        **{
                            name: frame[name]
                            for name in (
                                "video_id",
                                "participant_id",
                                "exercise",
                                "camera_view",
                                "source_video",
                                "error_type",
                                "pose_detected",
                            )
                        },
                        "label": label,
                        "frame_id": f"{label}_{frame_index:02d}",
                        "timestamp_ms": frame_index * 200,
                        **features,
                    }
                )
        windows, invalid = build_temporal_windows(pd.DataFrame(rows), "plank")
        self.assertEqual(set(windows["label"]), {"correct", "incorrect"})
        self.assertEqual(len(windows), 2)
        self.assertFalse(invalid)
        self.assertTrue(
            (windows["shoulder_hip_ankle_angle_valid_frames"] >= 5).all()
        )


if __name__ == "__main__":
    unittest.main()
