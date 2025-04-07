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
    remove_small_components,
    generate_postprocessing_labelmap,
    generate_processing_labelmap,
    apply_dynamic_threshold,
    convert_to_grayscale,
    invert_image_binary,
)


class CombinedWorker:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(script_dir, "../config.json"))

        self.image_path = os.path.abspath(os.path.join(script_dir, "../../example_images"))
        self.weights_path = os.path.abspath(os.path.join(script_dir, Path(__file__).resolve().parent.parent.parent / "weights" / "default.pth"))

        with open(config_path, 'r') as config_file:
            self.labels = json.load(config_file)["Labels"]

        self.confidence = 30
        self.manual_threshold = 180
        self.alveoli_minimum_size = 250
        self.parenchyma_minimum_size = 250
        self.number_of_lines = 3
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

    def set_weights_path(self, weights):
        self.weights_path = weights

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_manual_threshold(self, manual_threshold):
        self.manual_threshold = manual_threshold

    def set_alveoli_minimum_size(self, alveoli_minimum_size):
        self.alveoli_minimum_size = alveoli_minimum_size

    def set_parenchyma_minimum_size(self, parenchyma_minimum_size):
        self.parenchyma_minimum_size = parenchyma_minimum_size

    def set_number_of_lines(self, lines):
        self.number_of_lines = lines

    def set_minimum_length(self, length):
        self.minimum_length = length

    def set_scale(self, scale):
        self.scale = scale

    def set_randomized_distribution(self, randomized_distribution):
        self.randomized_distribution = randomized_distribution

    def run_processing(self):
        if not self.image_path:
            raise ValueError("[-] Error: Image path is not set.")

        if self.confidence is None:
            raise ValueError("[-] Error: Confidence is not set.")

        try:
            self.rgb_image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)[:, :, ::-1]
            model = model_operations.init_trained_model(self.weights_path)

            model_output = model_operations.run_prediction(self.image_path, model)
            self.inference_labelmap = generate_processing_labelmap(model_output, self.rgb_image.shape, self.confidence,
                                                                   self.labels)
        except Exception as e:
            print(f"[-] Error in processing: {e}")

    def run_postprocessing(self):
        if self.rgb_image is None:
            raise ValueError("[-] Error: Run processing first")

        if self.parenchyma_minimum_size is None:
            raise ValueError("[-] Error: Parenchyma minimum size is not set")

        if self.alveoli_minimum_size is None:
            raise ValueError("[-] Error: Alveoli minimum size is not set")

        if self.inference_labelmap is None:
            raise ValueError("[-] Error: Run processing first")

        try:
            grayscaled = convert_to_grayscale(self.rgb_image)
            thresholded = apply_dynamic_threshold(grayscaled)
            parenchyma_cleaned = remove_small_components(thresholded, self.parenchyma_minimum_size)
            inverted = invert_image_binary(parenchyma_cleaned)
            alveoli_cleaned = remove_small_components(inverted, self.alveoli_minimum_size)
            inverted_back = invert_image_binary(alveoli_cleaned)

            self.labelmap = generate_postprocessing_labelmap(self.inference_labelmap, inverted_back, self.labels)
        except Exception as e:
            print(f"[-] Error: Error in post-processing: {e}")

    def run_assessments(self):
        if self.labelmap is None:
            raise ValueError("[-] Error: Run postprocessing first")

        if self.number_of_lines is None:
            raise ValueError("[-] Error: Lines is not set")

        if self.minimum_length is None:
            raise ValueError("[-] Error: Length is not set")

        if self.scale is None:
            raise ValueError("[-] Error: Scale is not set")

        if not self.image_path:
            raise ValueError("[-] Error: Image path is not set")

        try:
            self.mli, self.assessments_layer, self.number_of_chords, self.stdev_chord_lengths = calculate_mean_linear_intercept(
                self.labelmap, self.number_of_lines, self.minimum_length, self.scale, self.labels, self.randomized_distribution)
            self.asvd, self.airspace_pixels, self.non_airspace_pixels = calculate_airspace_volume_density(self.labelmap,
                                                                                                          self.labels)
            self.shortened_image_path = os.path.join(os.path.basename(os.path.dirname(self.image_path)),
                                                     os.path.basename(self.image_path))

            self.current_results = (
                self.shortened_image_path, self.weights_path, self.asvd, self.mli, self.stdev_chord_lengths,
                self.number_of_chords, self.airspace_pixels, self.non_airspace_pixels, self.number_of_lines, self.minimum_length,
                self.scale)

            self.accumulated_results.append(self.current_results)
        except Exception as e:
            print(f"[-] Error in metrics calculation: {e}")

    def run_complete_pipline(self):
        self.run_processing()
        self.run_postprocessing()
        self.run_assessments()

    def get_current_results(self):
        if not self.current_results:
            print("[!] Warning: Current results is None")

        return self.current_results

    def get_accumulated_results(self):
        if not self.accumulated_results:
            print("[!] Warning: Accumulated results is None")

        return self.accumulated_results

    def get_shortened_image_path(self):
        if not self.shortened_image_path:
            print("[!] Warning: Shortened image path is None")

        return self.shortened_image_path

    def get_asvd(self):
        if self.asvd is None:
            print("[!] Warning: ASVD is None")

        return self.asvd

    def get_mli(self):
        if self.mli is None:
            print("[!] Warning: MLI is None")

        return self.mli

    def get_stdev_chord_lengths(self):
        if self.stdev_chord_lengths is None:
            print("[!] Warning: Standard deviation of chord lengths is None")

        return self.stdev_chord_lengths

    def get_number_of_chords(self):
        if self.number_of_chords is None:
            print("[!] Warning: Chords is None")

        return self.number_of_chords

    def get_airspace_pixels(self):
        if self.airspace_pixels is None:
            print("[!] Warning: Airspace pixels is None")

        return self.airspace_pixels

    def get_non_airspace_pixels(self):
        if self.non_airspace_pixels is None:
            print("[!] Warning: Non-airspace pixels is None")

        return self.non_airspace_pixels

    def get_lines(self):
        if self.number_of_lines is None:
            print("[!] Warning: Lines is None")

        return self.number_of_lines

    def get_length(self):
        if self.minimum_length is None:
            print("[!] Warning: Length is None")

        return self.minimum_length

    def get_scale(self):
        if self.scale is None:
            print("[!] Warning: Scale is None")

        return self.scale
