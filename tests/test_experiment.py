"""Tests for grouped experiment model candidates."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.experiment import _candidate_models, _positive_scores


class ExperimentCandidateTests(unittest.TestCase):
    def test_xgboost_candidate_handles_project_string_labels(self):
        estimator, distributions, requires_scaling = _candidate_models("plank")["xgboost"]
        features = pd.DataFrame(
            {
                "feature_a": np.linspace(-1.0, 1.0, 20),
                "feature_b": np.tile([0.0, 1.0], 10),
            }
        )
        labels = pd.Series(["correct", "incorrect"] * 10)

        fitted = clone(estimator).set_params(
            classifier__estimator__n_estimators=5
        ).fit(features, labels)
        predictions = fitted.predict(features)
        scores = _positive_scores(fitted, features)

        self.assertIn("classifier__estimator__max_depth", distributions)
        self.assertFalse(requires_scaling)
        self.assertTrue(set(predictions).issubset({"correct", "incorrect"}))
        self.assertEqual(scores.shape, (len(features),))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
