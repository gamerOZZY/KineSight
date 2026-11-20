"""Unit tests for reusable 3D geometry calculations."""

import unittest

import numpy as np

from src.features.geometry import (
    angle_degrees,
    center_point,
    distance_3d,
    point_to_line_distance_3d,
    point_to_line_vertical_deviation_3d,
)


class GeometryTests(unittest.TestCase):
    def test_distance_3d(self):
        self.assertAlmostEqual(distance_3d((0, 0, 0), (3, 4, 0)), 5.0)

    def test_angle_uses_dot_product_and_clips_cosine(self):
        self.assertAlmostEqual(angle_degrees((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90.0)

    def test_angle_rejects_zero_length_arm(self):
        self.assertIsNone(angle_degrees((0, 0, 0), (0, 0, 0), (1, 0, 0)))

    def test_point_to_line_distance_3d(self):
        self.assertAlmostEqual(
            point_to_line_distance_3d((2, 3, 4), (0, 0, 0), (1, 0, 0)),
            5.0,
        )

    def test_vertical_deviation_from_line(self):
        self.assertAlmostEqual(
            point_to_line_vertical_deviation_3d((0.5, 2, 0), (0, 0, 0), (1, 0, 0)),
            2.0,
        )

    def test_degenerate_line_is_invalid(self):
        self.assertIsNone(point_to_line_distance_3d((1, 2, 3), (0, 0, 0), (0, 0, 0)))

    def test_center_point_requires_all_finite_points(self):
        np.testing.assert_allclose(center_point([(0, 0, 0), (2, 2, 2)]), (1, 1, 1))
        self.assertIsNone(center_point([(0, 0, 0), None]))


if __name__ == "__main__":
    unittest.main()
