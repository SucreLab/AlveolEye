import unittest
import numpy as np
import torch
from alveoleye.lungcv.postprocessor import *


class TestThresholdMethods(unittest.TestCase):
    def setUp(self):
        # Define multiple sample greyscale images for testing
        self.test_cases = [
            {
                'input': np.array([], dtype=np.uint8),
                'expected_result_dynamic': np.array([], dtype=np.uint8),
                'expected_result_manual': np.array([], dtype=np.uint8)
            },
            {
                'input': np.array([[0, 0, 0, 0],
                                   [255, 255, 7, 19],
                                   [37, 97, 100, 198],
                                   [50, 190, 200, 200]], dtype=np.uint8),
                'expected_result_dynamic': np.array([[0, 0, 0, 0],
                                                     [255, 255, 0, 0],
                                                     [0, 0, 0, 255],
                                                     [0, 255, 255, 255]], dtype=np.uint8),
                'expected_result_manual': np.array([[0, 0, 0, 0],
                                                    [255, 255, 0, 255],
                                                    [255, 255, 255, 255],
                                                    [255, 255, 255, 255]], dtype=np.uint8)
            },
        ]

    def test_dynamic_threshold(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                if case['input'] is None:
                    # Test if the function raises an AssertionError for None input
                    with self.assertRaises(AssertionError):
                        dynamic_threshold(case['input'])
                else:
                    result = dynamic_threshold(case['input'])
                    expected_result = case['expected_result_dynamic']
                    np.testing.assert_array_equal(result, expected_result)

    def test_manual_threshold(self):
        for case in self.test_cases:
            with self.subTest(case=case):
                if case['input'] is None:
                    # Test if the function raises an AssertionError for None input
                    with self.assertRaises(AssertionError):
                        manual_threshold(case['input'], 10)
                else:
                    result = manual_threshold(case['input'], 10)
                    expected_result = case['expected_result_manual']
                    # Compare if expected_result and the result are equal
                    np.testing.assert_array_equal(result, expected_result)


class TestInvertBinary(unittest.TestCase):
    def setUp(self):
        # Create sample binary image for testing
        self.invert_binary_test = np.array([[255, 255, 0, 0, 0],
                                            [255, 0, 0, 0, 255],
                                            [0, 255, 255, 0, 0],
                                            [255, 0, 0, 0, 0],
                                            [255, 255, 255, 255, 0]], dtype=np.uint8)

    def test_invert_binary(self):
        self.assertIsNotNone(self.invert_binary_test)
        np.testing.assert_array_equal(np.unique(self.invert_binary_test), [0, 255])


class TestCleanMethodOne(unittest.TestCase):
    def setUp(self):
        # Create a sample binary image for testing
        self.binary_image = np.array([[0, 255, 0, 0, 0, 0, 0, 0],
                                      [0, 255, 255, 0, 0, 0, 255, 0],
                                      [0, 255, 255, 0, 0, 0, 255, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 255, 0, 0, 255, 255, 255, 0],
                                      [0, 255, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    def test_clean(self):
        # Call the clean function
        result_one = clean(self.binary_image.copy(), 4)
        result_two = clean(self.binary_image.copy(), 2)

        # Checks that binary_image is actually binary
        unique_values_one = np.unique(result_one)
        unique_values_two = np.unique(result_two)
        expected_values = [0, 255]

        np.testing.assert_array_equal(unique_values_one, expected_values)
        np.testing.assert_array_equal(unique_values_two, expected_values)

        # Manually create the expected results
        expected_result_one = np.array([[0, 255, 0, 0, 0, 0, 0, 0],
                                        [0, 255, 255, 0, 0, 0, 0, 0],
                                        [0, 255, 255, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        expected_result_two = np.array([[0, 255, 0, 0, 0, 0, 0, 0],
                                        [0, 255, 255, 0, 0, 0, 0, 0],
                                        [0, 255, 255, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 255, 255, 255, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        # Assert the result matches the expected result
        np.testing.assert_array_equal(result_one, expected_result_one)
        np.testing.assert_array_equal(result_two, expected_result_two)


class TestCleanMethodTwo(unittest.TestCase):
    def test_empty_image(self):
        binary_image = np.zeros((5, 5), dtype=np.uint8)
        result = clean(binary_image, 1)
        np.testing.assert_array_equal(result, binary_image)

    def test_full_image(self):
        binary_image = np.ones((5, 5), dtype=np.uint8)
        result = clean(binary_image, 1)
        np.testing.assert_array_equal(result, binary_image)

    def test_minimum_size_edge(self):
        binary_image = np.zeros((5, 5), dtype=np.uint8)
        binary_image[2, 2] = 1
        result = clean(binary_image, 1)
        # The single pixel should be removed, resulting in an all-zero image
        expected_output = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_output)

    def test_single_pixel_components(self):
        binary_image = np.zeros((5, 5), dtype=np.uint8)
        binary_image[1, 1] = 1
        binary_image[3, 3] = 1
        result = clean(binary_image, 1)

        # Expect both single pixel components be removed
        expected_output = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_output)

    def test_large_minimum_size(self):
        # Test case where the minimum size is larger than the entire image
        binary_image = np.ones((5, 5), dtype=np.uint8)
        result = clean(binary_image, 25)

        # Expect all pixels to be removed, resulting in an all-zero image
        expected_output = np.zeros((5, 5), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_output)

    def test_negative_minimum_size(self):
        binary_image = np.ones((5, 5), dtype=np.uint8)
        result = clean(binary_image, -1)
        np.testing.assert_array_equal(result, binary_image)


class TestCreateClassLabelmapFromModelOne(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.model_output = {
            "masks": [
                torch.tensor([[0.1, 0.2, 0.9], [0.4, 0.8, 0.6], [0.7, 0.3, 0.5]]),
                torch.tensor([[0.9, 0.7, 0.2], [0.3, 0.5, 0.4], [0.6, 0.8, 0.1]]),
                torch.tensor([[0.3, 0.6, 0.8], [0.2, 0.4, 0.7], [0.5, 0.1, 0.9]]),
                torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.9, 0.0]])
            ],
            "labels": torch.tensor([1, 2, 1, 1]),
            "scores": torch.tensor([0.95, 0.85, 0.75, 0.8])
        }
        self.class_id = 1
        self.confidence_threshold = 0.7

    def test_create_class_labelmap_from_model(self):
        result = create_class_labelmap_from_model(self.model_output, self.class_id, self.confidence_threshold)
        expected_output = np.array([[False, False, True], [False, True, False], [False, True, True]])

        np.testing.assert_array_equal(result, expected_output)


class TestCreateClassLabelmapFromModelTwo(unittest.TestCase):
    def test_empty_model_output(self):
        model_output = {"masks": [], "labels": [], "scores": []}

        # Call the function with the empty model output
        result = create_class_labelmap_from_model(model_output, 1, 0.5)

        # Expect an empty array as output
        expected_output = np.array([], dtype=bool)
        np.testing.assert_array_equal(result, expected_output)

    def test_no_masks_matching_class_id(self):
        model_output = {
            "masks": torch.tensor([[[0, 1], [1, 0]]]),
            "labels": torch.tensor([2]),
            "scores": torch.tensor([0.8])
        }
        result = create_class_labelmap_from_model(model_output, 1, 0.5)

        # Expect an array of False values--no masks match the class ID
        expected_output = np.array([[False, False], [False, False]])
        np.testing.assert_array_equal(result, expected_output)

    def test_no_masks_above_confidence_threshold(self):
        model_output = {
            "masks": torch.tensor([[[0, 1], [1, 0]]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.3])
        }
        result = create_class_labelmap_from_model(model_output, class_id=1, confidence_threshold=0.5)

        # Expect an array of False values--no masks meet the confidence threshold
        expected_output = np.array([[False, False], [False, False]])
        np.testing.assert_array_equal(result, expected_output)

    def test_single_mask_matching_criteria(self):
        model_output = {
            "masks": torch.tensor([[[0, 1], [1, 0]]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.8])
        }
        result = create_class_labelmap_from_model(model_output, class_id=1, confidence_threshold=0.5)
        expected_output = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(result, expected_output)

    def test_multiple_masks_matching_criteria(self):
        model_output = {
            "masks": [
                torch.tensor([[0, 1], [1, 0]]),
                torch.tensor([[1, 0], [0, 1]])
            ],
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.8, 0.7])
        }
        result = create_class_labelmap_from_model(model_output, class_id=1, confidence_threshold=0.5)

        # Expect the masks to be combined in the result
        expected_output = np.array([[True, True], [True, True]])
        np.testing.assert_array_equal(result, expected_output)

    def test_varying_dimensions_of_masks(self):
        model_output = {
            "masks": [
                torch.tensor([[0, 1], [1, 0]]),
                torch.tensor([[1, 0, 1], [0, 1, 0]])
            ],
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.8, 0.7])
        }

        # Expect ValueError
        with self.assertRaises(ValueError):
            create_class_labelmap_from_model(model_output, class_id=1, confidence_threshold=0.5)


class TestCreateProcessingLabelmap(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.model_output = {
            "masks": [
                torch.tensor([[0.1, 0.2, 0.9], [0.4, 0.8, 0.6], [0.7, 0.3, 0.5]]),
                torch.tensor([[0.9, 0.7, 0.2], [0.3, 0.5, 0.4], [0.6, 0.8, 0.1]]),
                torch.tensor([[0.3, 0.6, 0.8], [0.2, 0.4, 0.7], [0.5, 0.1, 0.9]])
            ],
            "labels": torch.tensor([1, 2, 1]),
            "scores": torch.tensor([0.95, 0.85, 0.75])
        }
        self.shape = (3, 3)
        self.confidence_threshold = 50  # 50%
        self.labels = {"AIRWAY_EPITHELIUM": 1, "VESSEL_ENDOTHELIUM": 2, "BLOCKER": 9}

    def test_create_processing_labelmap(self):
        # Test case for creating a labelmap with standard parameters
        expected_output = np.array([[2, 2, 1], [0, 1, 1], [2, 2, 1]], dtype=np.uint8)
        result = create_processing_labelmap(self.model_output, self.shape, self.confidence_threshold, self.labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_high_confidence_threshold(self):
        high_confidence_threshold = 90
        expected_output = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        result = create_processing_labelmap(self.model_output, self.shape, high_confidence_threshold, self.labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_low_confidence_threshold(self):
        low_confidence_threshold = 10
        expected_output = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 1]], dtype=np.uint8)
        result = create_processing_labelmap(self.model_output, self.shape, low_confidence_threshold, self.labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_empty_masks(self):
        model_output = {
            "masks": [],
            "labels": torch.tensor([]),
            "scores": torch.tensor([])
        }
        expected_output = np.zeros(self.shape, dtype=np.uint8)
        result = create_processing_labelmap(model_output, self.shape, self.confidence_threshold, self.labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_different_labels(self):
        different_labels = {"AIRWAY_EPITHELIUM": 3, "VESSEL_ENDOTHELIUM": 4}
        expected_output = np.array([[4, 4, 3], [0, 3, 3], [4, 4, 3]], dtype=np.uint8)
        result = create_processing_labelmap(self.model_output, self.shape, self.confidence_threshold, different_labels)
        np.testing.assert_array_equal(result, expected_output)

    def test_non_standard_shape(self):
        non_standard_shape = (2, 3)

        # Expect ValueError--non-standard shape
        with self.assertRaises(ValueError):
            create_processing_labelmap(self.model_output, non_standard_shape, self.confidence_threshold, self.labels)


class TestCreateCompleteClassLabelmap(unittest.TestCase):
    def setUp(self):
        # Define the labels
        self.epithelium_label = 1
        self.lumen_label = 2

        # Create the class_epithelium_labelmap array
        self.class_epithelium_labelmap = np.zeros((50, 50), dtype=np.uint8)
        self.class_epithelium_labelmap[5:15, 5:15] = self.epithelium_label
        self.class_epithelium_labelmap[30:40, 30:40] = self.epithelium_label
        self.class_epithelium_labelmap[20:25, 20:25] = self.lumen_label

        for i in range(10):
            self.class_epithelium_labelmap[i + 10, i + 10] = self.epithelium_label

        # Create a thresholded image (binary image)
        self.thresholded_image = np.zeros((50, 50), dtype=np.uint8)
        self.thresholded_image[20:25, 20:25] = 1  # Lumen region

    def test_create_complete_class_labelmap(self):
        result = create_complete_class_labelmap(self.class_epithelium_labelmap, self.thresholded_image,
                                                self.epithelium_label, self.lumen_label)

        allowed_values = {0, 1, 2}
        unique_values = np.unique(result)

        for value in unique_values:
            self.assertIn(value, allowed_values, f"Value {value} is not allowed in the array")

        self.assertTrue(np.all(result[:, 0] == 0))
        self.assertFalse(np.all(result[:, -1] == 1))
        self.assertFalse(np.all(result[0, :] == 1))
        self.assertFalse(np.all(result[-1, :] == 1))

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (50, 50))  # Ensure output shape matches input shape
        self.assertTrue(np.any(result == self.epithelium_label))


class TestPostprocessingLabelmapOne(unittest.TestCase):

    def setUp(self):
        # Define labels
        self.labels = {
            "AIRWAY_EPITHELIUM": 1,
            "VESSEL_ENDOTHELIUM": 2,
            "AIRWAY_LUMEN": 3,
            "VESSEL_LUMEN": 4,
            "ALVEOLI": 5,
            "PARENCHYMA": 6,
            "BLOCKER": 9
        }

        # Base mask for testing
        self.masks_labelmap = np.zeros((50, 50), dtype=np.uint8)
        self.thresholded_labelmap = np.zeros((50, 50), dtype=np.uint8)

    def test_no_labels(self):
        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)
        self.assertTrue(np.all(result == self.labels["PARENCHYMA"]))

    def test_full_thresholded_labelmap(self):
        self.thresholded_labelmap[:, :] = 1
        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)
        self.assertTrue(np.all(result == self.labels["ALVEOLI"]))

    def test_edge_touching_epithelium(self):
        self.masks_labelmap[0, :] = self.labels["AIRWAY_EPITHELIUM"]
        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)
        self.assertTrue(np.all(result[0, :] == self.labels["AIRWAY_EPITHELIUM"]))

    def test_internal_epithelium(self):
        self.masks_labelmap[10:20, 10:20] = self.labels["AIRWAY_EPITHELIUM"]
        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)
        self.assertTrue(np.all(result[10:20, 10:20] == self.labels["AIRWAY_EPITHELIUM"]))

    def test_internal_lumen(self):
        self.thresholded_labelmap[15:17, 15:17] = 1
        self.masks_labelmap[10:20, 10:20] = self.labels["VESSEL_ENDOTHELIUM"]

        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)

        self.assertFalse(np.any(result[15:17, 15:17] == self.labels["VESSEL_ENDOTHELIUM"]))

    def test_combined_case(self):
        # Setup labelmap regions for testing
        self.masks_labelmap[5:15, 5:15] = self.labels["AIRWAY_EPITHELIUM"]
        self.masks_labelmap[30:40, 30:40] = self.labels["VESSEL_ENDOTHELIUM"]
        self.thresholded_labelmap[35:37, 35:37] = 1  # Thresholded alveoli region
        self.thresholded_labelmap[0, :] = 1  # Thresholded top row

        # Generate postprocessed labelmap
        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)

        # Check parenchyma
        self.assertTrue(np.any(result[1:, 1:] != self.labels["PARENCHYMA"]))

        # Check alveoli
        self.assertTrue(np.all(result[35:37, 35:37] == self.labels["ALVEOLI"]))

        result[35:37, 35:37] = 42  # Temporarily replace ALVEOLI region
        self.assertFalse(np.any(result[1:, 1:] == self.labels["ALVEOLI"]))

        # Check vessel endothelium labels after excluding ALVEOLI region
        result[35:37, 35:37] = self.labels["VESSEL_ENDOTHELIUM"]  # Temporarily replace ALVEOLI region
        self.assertTrue(np.all(result[30:40, 30:40] == self.labels["VESSEL_ENDOTHELIUM"]))

        # Check airway epithelium
        self.assertTrue(np.all(result[5:15, 5:15] == self.labels["AIRWAY_EPITHELIUM"]))


class TestCreatePostprocessingLabelmapTwo(unittest.TestCase):
    def setUp(self):
        # Setup test data
        self.masks_labelmap = np.array([[0, 3], [4, 2]], dtype=np.uint8)
        self.thresholded_labelmap = np.array([[255, 0], [255, 0]], dtype=np.uint8)
        self.labels = {
            "ALVEOLI": 1,
            "PARENCHYMA": 2,
            "AIRWAY_EPITHELIUM": 3,
            "VESSEL_ENDOTHELIUM": 4,
            "AIRWAY_LUMEN": 5,
            "VESSEL_LUMEN": 6,
            "BLOCKER": 9
        }

    def test_create_postprocessing_labelmap(self):
        result = create_postprocessing_labelmap(self.masks_labelmap, self.thresholded_labelmap, self.labels)

        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check that the result has the same shape as the input masks_labelmap
        self.assertEqual(result.shape, self.masks_labelmap.shape)

        # Check that the result contains only expected label values
        expected_labels = set(self.labels.values()).union({0})
        unique_labels_in_result = set(np.unique(result))
        self.assertTrue(unique_labels_in_result.issubset(expected_labels))


if __name__ == "__main__":
    unittest.main()