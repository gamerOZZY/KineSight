"""Tests for safe pairing of exercise videos, metadata, and frame directories."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pose.landmarks import discover_video_records, source_frame_index


class VideoDiscoveryTests(unittest.TestCase):
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
