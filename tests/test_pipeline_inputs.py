"""Tests for safe pairing of exercise videos, metadata, and frame directories."""

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.landmarks import discover_video_records, source_frame_index
from src.preprocessing.preparation import create_exercise_metadata


class VideoDiscoveryTests(unittest.TestCase):
    def test_metadata_creation_normalizes_hollow_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = (
                root
                / "data"
                / "bronze"
                / "videos"
                / "hollow_body"
                / "correct"
                / "good_hollow_01_p3_bottom-side.mp4"
            )
            video.parent.mkdir(parents=True)
            video.touch()
            with patch(
                "src.preprocessing.preparation._video_properties",
                return_value=(240, 30.0),
            ):
                metadata, metadata_path = create_exercise_metadata(root, "hollow_body")

            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata.iloc[0]["excercise"], "hollow_body")
            self.assertEqual(metadata.iloc[0]["Participant_id"], "P3")
            self.assertEqual(metadata.iloc[0]["camera_view"], "bottom-side")
            self.assertEqual(metadata.iloc[0]["duration_in_seconds"], 8.0)
            self.assertTrue(metadata_path.is_file())

    def test_exercise_name_with_underscore_is_parsed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video = (
                root
                / "data"
                / "bronze"
                / "videos"
                / "hollow_body"
                / "correct"
                / "good_hollow_body_01_p3_back-side.mp4"
            )
            video.parent.mkdir(parents=True)
            video.touch()
            (root / "data/silver/extracted_frames/hollow_body/P01").mkdir(
                parents=True
            )
            metadata = root / "data/bronze/metadata/hollow_body_metadata.csv"
            metadata.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "video_id": "P01",
                        "Participant_id": "P3",
                        "excercise": "hollow_body",
                        "label": "correct",
                        "camera_view": "back-side",
                    }
                ]
            ).to_csv(metadata, index=False)

            records = discover_video_records(root, "hollow_body")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].participant_id, "P3")
            self.assertEqual(records[0].exercise, "hollow_body")
            self.assertEqual(records[0].source_video, video)

    def test_source_index_matches_seven_fps_sampling(self):
        self.assertEqual(
            [source_frame_index(index, 30.0, 7.0) for index in range(6)],
            [0, 5, 9, 13, 18, 22],
        )
        self.assertEqual(
            [source_frame_index(index, 5.0, 7.0) for index in range(4)],
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
