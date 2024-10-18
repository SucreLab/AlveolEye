import unittest
import numpy as np
import cv2
from alveoleye.lungcv.assessments import *


class TestCalculateAirspaceVolumeDensity(unittest.TestCase):
    def setUp(self):
        # Set up a common label dictionary for testing
        self.labels = {
            "AIRWAY_LUMEN": 1,
            "VESSEL_LUMEN": 2,
            "AIRWAY_EPITHELIUM": 3,
            "VESSEL_ENDOTHELIUM": 4,
            "ALVEOLI": 5,
            "BLOCKER": 9
        }

    def test_basic_case(self):
        # Simple case with known values
        labelmap = np.array([
            [1, 1, 1, 5],
            [1, 5, 2, 2],
            [5, 5, 3, 3],
            [5, 5, 4, 4]
        ])
        expected_density = 100.00  # 6 alveoli pixels out of 8 non-lumen pixels
        self.assertEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], expected_density)

    def test_no_alveoli(self):
        # Case with no alveoli pixels
        labelmap = np.array([
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]
        ])
        expected_density = 0.0  # 0 alveoli pixels

        if np.sum(labelmap == 5) == 0:
            with self.assertRaises(ZeroDivisionError):
                calculate_airspace_volume_density(labelmap, self.labels)
        else:
            self.assertAlmostEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], expected_density)

    def test_all_alveoli(self):
        # Case with all pixels as alveoli
        labelmap = np.array([
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5]
        ])
        expected_density = 100.0  # All are alveoli pixels
        self.assertAlmostEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], expected_density)

    def test_mixed_case(self):
        # More complex case with mixed labels
        labelmap = np.array([
            [1, 2, 5, 5],
            [3, 4, 1, 5],
            [5, 2, 1, 5],
            [5, 5, 3, 4]
        ])
        expected_density = (7 / 7) * 100  # 7 alveoli pixels out of 8 non-lumen pixels
        self.assertAlmostEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], expected_density)


class TestCalculateMeanLinearIntercept(unittest.TestCase):
    def setUp(self):
        self.labels = {
            "ALVEOLI": 1,
            "MLI_LINES_OUTSIDE": 2,
            "MLI_LINES_INSIDE": 3,
            "BLOCKER": 9
        }

    def test_empty_labelmap(self):
        labelmap = np.array([[]], dtype=np.uint8)

        # Expect IndexError--empty labelmap
        with self.assertRaises(IndexError):
            calculate_mean_linear_intercept(labelmap, 0, 0, 1, self.labels)

    def test_single_row_labelmap(self):
        labelmap = np.array([[0, 0, 0, 0]], dtype=np.uint8)

        # Expect IndexError--one-dimensional labelmap
        with self.assertRaises(IndexError):
            calculate_mean_linear_intercept(labelmap, 2, 1, 1, self.labels)

    def test_single_column_labelmap(self):
        labelmap = np.array([[0], [0], [0], [0]], dtype=np.uint8)

        # Expect IndexError--one-dimensional labelmap
        with self.assertRaises(IndexError):
            calculate_mean_linear_intercept(labelmap, 2, 1, 1, self.labels)

    def test_all_zero_labelmap(self):
        labelmap = np.zeros((3, 3), dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 1, 1, 1, self.labels)
        expected_length = 0
        expected_labelmap = np.zeros((3, 3), dtype=int)
        self.assertAlmostEqual(result[0], expected_length)
        np.testing.assert_array_equal(result[1], expected_labelmap)

    def test_no_features_to_analyze(self):
        labelmap = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 2, 10, 1, self.labels)
        expected_length = 0
        expected_labelmap = np.zeros((3, 3), dtype=int)
        self.assertAlmostEqual(result[0], expected_length)
        np.testing.assert_array_equal(result[1], expected_labelmap)

    def test_single_component_larger_than_min_length(self):
        labelmap = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 2, 5, 1, self.labels)
        expected_length = 0  # Total area of the component
        self.assertAlmostEqual(result[0], expected_length)

    def test_multiple_components(self):
        labelmap = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 2, 2, 1, self.labels)
        self.assertGreater(result[0], 0)  # Expecting some non-zero average length
        self.assertTrue(np.any(result[1] == self.labels["MLI_LINES_INSIDE"]))  # Should have MLI_LINES_INSIDE label


if __name__ == '__main__':
    unittest.main()
