"""Verify each final model artifact matches its Gold feature schema."""

import json
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TrainedModelArtifactTests(unittest.TestCase):
    def test_exercise_models_load_and_predict_with_saved_feature_order(self):
        for exercise in ("plank", "hollow_body"):
            model_directory = PROJECT_ROOT / "models" / exercise
            metadata = json.loads(
                (model_directory / "model_metadata.json").read_text(encoding="utf-8")
            )
            model = joblib.load(model_directory / "model.joblib")
            dataset = pd.read_parquet(
                PROJECT_ROOT / "data" / "gold" / f"{exercise}_features.parquet"
            )
            columns = metadata["features"]

            self.assertEqual(metadata["exercise"], exercise)
            self.assertEqual(list(model.feature_names_in_), columns)
            predictions = model.predict(dataset[columns].head(2))
            self.assertEqual(len(predictions), min(2, len(dataset)))
            self.assertTrue(set(predictions).issubset({"correct", "incorrect"}))


if __name__ == "__main__":
    unittest.main()
