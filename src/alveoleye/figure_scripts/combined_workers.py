import json
import os
from pathlib import Path
import cv2

from alveoleye.lungcv import model_operations
from alveoleye.lungcv.assessments import (
    calculate_airspace_volume_density,
    calculate_mean_linear_intercept,
)
from alveoleye.lungcv.postprocessor import (
    clean,
    create_postprocessing_labelmap,
    create_processing_labelmap,
    dynamic_threshold,
    greyscale,
    invert_binary,
)


class CombinedWorker:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(script_dir, "../config.json"))

        self.image_path = os.path.abspath(os.path.join(script_dir, "../data"))
        self.weights_path = os.path.abspath(os.path.join(script_dir, "../../../data/default.pth"))

        with open(config_path, 'r') as config_file:
            self.labels = json.load(config_file)["Labels"]

        self.confidence = 30
        self.manual_threshold = 180
        self.alveoli_minimum_size = 250
        self.parenchyma_minimum_size = 250
        self.lines = 3
        self.minimum_length = 20
        self.scale = 0.18872
        self.randomized_distribution = False

        self.shortened_image_path = None
        self.asvd = None
        self.mli = None
        self.inference_labelmap = None
        self.labelmap = None
        self.mli = None
        self.asvd = None
        self.data = None
        self.airspace_pixels = None
        self.non_airspace_pixels = None
        self.assessments_layer = None
        self.number_of_chords = None
        self.stdev_chord_lengths = None
        self.rgb_image = None
        self.counter = 0
        self.first = None
        self.current_results = None
        self.accumulated_results = []

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_weights(self, weights):
        self.weights_path = weights

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_manual_threshold(self, manual_threshold):
        self.manual_threshold = manual_threshold

    def set_alveoli_minimum_size(self, alveoli_minimum_size):
        self.alveoli_minimum_size = alveoli_minimum_size

    def set_parenchyma_minimum_size(self, parenchyma_minimum_size):
        self.parenchyma_minimum_size = parenchyma_minimum_size

    def set_lines(self, lines):
        self.lines = lines

    def set_minimum_length(self, length):
        self.minimum_length = length

    def set_scale(self, scale):
        self.scale = scale

    def set_randomized_distribution(self, randomized_distribution):
        self.randomized_distribution = randomized_distribution

    def run_processing(self):
        if not self.image_path:
            raise ValueError("Image path is not set.")

        if not self.weights_path:
            raise ValueError("Weights path is not set.")

        if not self.confidence:
            raise ValueError("Confidence is not set.")

        try:
            self.rgb_image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)[:, :, ::-1]
            model = model_operations.init_trained_model(Path(self.weights_path))

            model_output = model_operations.run_prediction(self.image_path, model)
            self.inference_labelmap = create_processing_labelmap(model_output, self.rgb_image.shape, self.confidence,
                                                                 self.labels)
        except Exception as e:
            print(f"Error in processing: {e}")

    def run_postprocessing(self):
        if self.rgb_image is None:
            raise ValueError("Run processing first")

        if not self.parenchyma_minimum_size:
            raise ValueError("Parenchyma minimum size is not set")

        if not self.alveoli_minimum_size:
            raise ValueError("Alveoli minimum size is not set")

        if self.inference_labelmap is None:
            raise ValueError("Run processing first")

        try:
            greyscaled = greyscale(self.rgb_image)
            thresholded = dynamic_threshold(greyscaled)
            parenchyma_cleaned = clean(thresholded, self.parenchyma_minimum_size)
            inverted = invert_binary(parenchyma_cleaned)
            alveoli_cleaned = clean(inverted, self.alveoli_minimum_size)
            inverted_back = invert_binary(alveoli_cleaned)

            self.labelmap = create_postprocessing_labelmap(self.inference_labelmap, inverted_back, self.labels)
        except Exception as e:
            print(f"Error in post-processing: {e}")

    def run_assessments(self):
        if self.labelmap is None:
            raise ValueError("Run postprocessing first")

        if not self.lines:
            raise ValueError("Lines is not set")

        if not self.minimum_length:
            raise ValueError("Length is not set")

        if not self.scale:
            raise ValueError("Scale is not set")

        if not self.image_path:
            raise ValueError("Image path is not set")

        try:
            self.mli, self.assessments_layer, self.number_of_chords, self.stdev_chord_lengths = calculate_mean_linear_intercept(
                self.labelmap, self.lines, self.minimum_length, self.scale, self.labels, self.randomized_distribution)
            self.asvd, self.airspace_pixels, self.non_airspace_pixels = calculate_airspace_volume_density(self.labelmap,
                                                                                                          self.labels)
            self.shortened_image_path = os.path.join(os.path.basename(os.path.dirname(self.image_path)),
                                                     os.path.basename(self.image_path))

            self.current_results = (
                self.shortened_image_path, self.weights_path, self.asvd, self.mli, self.stdev_chord_lengths,
                self.number_of_chords, self.airspace_pixels, self.non_airspace_pixels, self.lines, self.minimum_length,
                self.scale)

            self.accumulated_results.append(self.current_results)
        except Exception as e:
            print(f"Error in metrics calculation: {e}")

    def run_complete_pipline(self):
        self.run_processing()
        self.run_postprocessing()
        self.run_assessments()

    def get_current_results(self):
        if not self.current_results:
            raise ValueError("Current results is None")

        return self.current_results

    def get_accumulated_results(self):
        if not self.accumulated_results:
            raise ValueError("Accumulated results is None")

        return self.accumulated_results

    def get_shortened_image_path(self):
        if not self.shortened_image_path:
            raise ValueError("Shortened image path is None")

    def get_asvd(self):
        if not self.asvd:
            raise ValueError("ASVD is None")

        return self.asvd

    def get_mli(self):
        if not self.mli:
            raise ValueError("MLI is None")

        return self.mli

    def get_stdev_chord_lengths(self):
        if not self.stdev_chord_lengths:
            raise ValueError("Standard deviation of chord lengths is None")

        return self.stdev_chord_lengths

    def get_number_of_chords(self):
        if not self.number_of_chords:
            raise ValueError("Chords is None")

        return self.number_of_chords

    def get_airspace_pixels(self):
        if not self.airspace_pixels:
            raise ValueError("Airspace pixels is None")

        return self.airspace_pixels

    def get_non_airspace_pixels(self):
        if not self.non_airspace_pixels:
            raise ValueError("Non-airspace pixels is None")

        return self.non_airspace_pixels

    def get_lines(self):
        if not self.lines:
            raise ValueError("Lines is None")

        return self.lines

    def get_length(self):
        if not self.minimum_length:
            raise ValueError("Length is None")

        return self.minimum_length

    def get_scale(self):
        if not self.scale:
            raise ValueError("Scale is None")

        return self.scale
