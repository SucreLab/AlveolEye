import unittest
import numpy as np
from alveoleye.lungcv.assessments import calculate_airspace_volume_density, calculate_mean_linear_intercept

class TestCalculateAirspaceVolumeDensity(unittest.TestCase):
    def setUp(self):
        # Setup commonly used labels for testing
        self.labels = {
            "AIRWAY_LUMEN": 1,
            "VESSEL_LUMEN": 2,
            "AIRWAY_EPITHELIUM": 3,
            "VESSEL_ENDOTHELIUM": 4,
            "ALVEOLI": 5,
            "BLOCKER": 9
        }

    def test_basic_case(self):
        # Test with simple, known values
        labelmap = np.array([
            [1, 1, 1, 5],
            [1, 5, 2, 2],
            [5, 5, 3, 3],
            [5, 5, 4, 4]
        ])
        expected_density = 100.00  # 6 alveoli pixels out of 8 non-lumen pixels
        self.assertEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], expected_density)

    def test_no_alveoli(self):
        # Test case with no alveoli pixels
        labelmap = np.array([
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]
        ])

        if np.sum(labelmap == self.labels["ALVEOLI"]) == 0:
            with self.assertRaises(ZeroDivisionError):
                calculate_airspace_volume_density(labelmap, self.labels)
        else:
            self.assertAlmostEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], 0.0)

    def test_all_alveoli(self):
        # Test case with all alveoli pixels
        labelmap = np.full((4, 4), self.labels["ALVEOLI"])
        expected_density = 100.0
        self.assertAlmostEqual(calculate_airspace_volume_density(labelmap, self.labels)[0], expected_density)

    def test_mixed_case(self):
        # Complex case with mixed labels
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
        # Setup commonly used labels for MLI testing
        self.labels = {
            "ALVEOLI": 1,
            "MLI_LINES_OUTSIDE": 2,
            "MLI_LINES_INSIDE": 3,
            "BLOCKER": 9
        }

    def test_empty_labelmap(self):
        labelmap = np.array([[]], dtype=np.uint8)

        with self.assertRaises(IndexError):
            calculate_mean_linear_intercept(labelmap, 0, 0, 1, self.labels)

    def test_single_row_labelmap(self):
        labelmap = np.array([[0, 0, 0, 0]], dtype=np.uint8)

        with self.assertRaises(IndexError):
            calculate_mean_linear_intercept(labelmap, 2, 1, 1, self.labels)

    def test_single_column_labelmap(self):
        labelmap = np.array([[0], [0], [0], [0]], dtype=np.uint8)

        with self.assertRaises(IndexError):
            calculate_mean_linear_intercept(labelmap, 2, 1, 1, self.labels)

    def test_all_zero_labelmap(self):
        labelmap = np.zeros((3, 3), dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 1, 1, 1, self.labels)
        expected_length = 0
        np.testing.assert_array_equal(result[1], np.zeros((3, 3), dtype=int))
        self.assertAlmostEqual(result[0], expected_length)

    def test_no_features_to_analyze(self):
        labelmap = np.zeros((3, 3), dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 2, 10, 1, self.labels)
        expected_length = 0
        np.testing.assert_array_equal(result[1], np.zeros((3, 3), dtype=int))
        self.assertAlmostEqual(result[0], expected_length)

    def test_single_component_larger_than_min_length(self):
        labelmap = np.ones((3, 3), dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 2, 5, 1, self.labels)
        expected_length = 0  # Expecting no valid intercepts based on the input
        self.assertAlmostEqual(result[0], expected_length)

    def test_multiple_components(self):
        labelmap = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8)
        result = calculate_mean_linear_intercept(labelmap, 2, 2, 1, self.labels)
        self.assertGreater(result[0], 0)  # Expecting non-zero average length
        self.assertTrue(np.any(result[1] == self.labels["MLI_LINES_INSIDE"]))  # Ensure lines inside are labeled


if __name__ == '__main__':
    unittest.main()
