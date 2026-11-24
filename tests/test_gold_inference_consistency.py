"""Verify Gold temporal features can be rebuilt with the inference aggregator."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.biomechanics import (
    HOLLOW_BODY_FEATURE_NAMES,
    PLANK_FEATURE_NAMES,
    extract_frame_features,
)
from src.features.temporal import STATISTICS, aggregate_feature_window


class GoldInferenceConsistencyTests(unittest.TestCase):
    def test_gold_window_matches_recomputed_silver_frames(self):
        for exercise, feature_names in (
            ("plank", PLANK_FEATURE_NAMES),
            ("hollow_body", HOLLOW_BODY_FEATURE_NAMES),
        ):
            gold_path = PROJECT_ROOT / "data" / "gold" / f"{exercise}_features.parquet"
            gold = pd.read_parquet(gold_path)
            usable = gold.dropna(
                subset=[
                    f"{feature}_{statistic}"
                    for feature in feature_names
                    for statistic in STATISTICS
                ]
            )
            self.assertFalse(usable.empty, f"No fully valid Gold window for {exercise}.")
            expected = usable.iloc[0]
            frame_ids = expected["frame_ids"]
            if isinstance(frame_ids, np.ndarray):
                frame_ids = frame_ids.tolist()

            silver_path = (
                PROJECT_ROOT
                / "data"
                / "silver"
                / "landmarks"
                / exercise
                / f"{expected['video_id']}.parquet"
            )
            silver = pd.read_parquet(silver_path)
            selected = silver[silver["frame_id"].isin(frame_ids)].sort_values("timestamp_ms")
            self.assertEqual(len(selected), len(frame_ids))

            samples = []
            for frame in selected.to_dict(orient="records"):
                features, _ = extract_frame_features(frame, exercise)
                samples.append(
                    {
                        "timestamp_ms": frame["timestamp_ms"],
                        **features,
                    }
                )
            actual, invalid = aggregate_feature_window(
                pd.DataFrame(samples),
                exercise,
                window_start_ms=int(expected["window_start_ms"]),
            )
            self.assertFalse(invalid)
            for feature in feature_names:
                for statistic in STATISTICS:
                    self.assertAlmostEqual(
                        actual[f"{feature}_{statistic}"],
                        expected[f"{feature}_{statistic}"],
                        places=8,
                        msg=f"{exercise}: {feature}_{statistic}",
                    )


if __name__ == "__main__":
    unittest.main()
