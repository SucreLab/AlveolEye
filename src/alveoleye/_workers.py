import numpy as np
from qtpy.QtCore import QObject, Signal

from alveoleye.lungcv import model_operations
from alveoleye.lungcv.postprocessor import (manual_threshold, dynamic_threshold,
                                                             create_postprocessing_labelmap, clean, greyscale,
                                                             invert_binary, create_processing_labelmap)
from alveoleye.lungcv.assessments import (calculate_mean_linear_intercept,
                                                           calculate_airspace_volume_density)
import pathlib
import json
import alveoleye._layers_editor as layers_editor
import alveoleye._export_operations as export_operations


class WorkerParent(QObject):
    finished = Signal()
    layers_config_data = None

    def __init__(self):
        super().__init__()
        self.napari_viewer = None
        self.layer_names = None
        self.labels = None
        self.terminate = False
        self.model = None
        with open(pathlib.Path(__file__).resolve().parent / "config.json", 'r') as config_file:
            self.config_data = json.load(config_file)

    def set_napari_viewer(self, napari_viewer):
        self.napari_viewer = napari_viewer

    def set_layer_names(self, layer_names):
        self.layer_names = layer_names

    def set_labels(self, labels):
        self.labels = labels

    def cancel(self):
        self.terminate = True


class ProcessingWorker(WorkerParent):
    results_ready = Signal(dict, np.ndarray)

    def __init__(self):
        super().__init__()
        self.image_path = None
        self.image_shape = None
        self.weights = None
        self.confidence_threshold_value = None

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_image_shape(self, image_shape):
        self.image_shape = image_shape

    def set_weights(self, weights):
        try:
            self.model = model_operations.init_trained_model(weights)
        except Exception as e:
            print(f"Error initializing model: {e}")

    def set_confidence_threshold_value(self, confidence_threshold_value):
        self.confidence_threshold_value = confidence_threshold_value

    def run(self):
        try:
            if self.model is None:
                # Try to load.. then fail
                try:
                    print("Loading default weights")
                    default_weights = self.config_data["ProcessingActionBox"]["DEFAULT_WEIGHTS_PATH"]
                    self.model = model_operations.init_trained_model(default_weights)
                except Exception as e:
                    print(f"Model has not been loaded, aborting prediction: {e}")
                    self.terminate = True

            if not self.terminate:
                model_output = model_operations.run_prediction(self.image_path, self.model)

            if not self.terminate:
                inference_labelmap = create_processing_labelmap(model_output, self.image_shape,
                                                                self.confidence_threshold_value, self.labels)

            if not self.terminate:
                self.results_ready.emit(model_output, inference_labelmap)
        except Exception as e:
            print(f"Error in processing: {e}")
        finally:
            self.finished.emit()


class PostprocessingWorker(WorkerParent):
    results_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.model_output = None
        self.thresholding_check_box_value = None
        self.manual_threshold_value = None
        self.alveoli_minimum_size = None
        self.parenchyma_minimum_size = None
        self.confidence_threshold = None

    def set_model_output(self, model_output):
        self.model_output = model_output

    def set_thresholding_check_box_value(self, thresholding_check_box_value):
        self.thresholding_check_box_value = thresholding_check_box_value

    def set_manual_threshold_value(self, manual_threshold_value):
        self.manual_threshold_value = manual_threshold_value

    def set_alveoli_minimum_size(self, alveoli_minimum_size):
        self.alveoli_minimum_size = alveoli_minimum_size

    def set_parenchyma_minimum_size(self, parenchyma_minimum_size):
        self.parenchyma_minimum_size = parenchyma_minimum_size

    def threshold_according_to_method(self, image):
        if self.thresholding_check_box_value:
            return manual_threshold(image, self.manual_threshold_value)

        return dynamic_threshold(image)

    def run(self):
        try:
            if not self.terminate:
                image = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["INITIAL_LAYER"])

            if not self.terminate:
                greyscaled = greyscale(image)

            if not self.terminate:
                thresholded = self.threshold_according_to_method(greyscaled)

            if not self.terminate:
                parenchyma_cleaned = clean(thresholded, self.parenchyma_minimum_size)

            if not self.terminate:
                inverted = invert_binary(parenchyma_cleaned)

            if not self.terminate:
                alveoli_cleaned = clean(inverted, self.alveoli_minimum_size)

            if not self.terminate:
                inverted_back = invert_binary(alveoli_cleaned)

            if not self.terminate:
                masks_labelmap = layers_editor.get_layer_by_name(self.napari_viewer,
                                                                 self.layer_names["PROCESSING_LAYER"])

            if not self.terminate:
                labelmap = create_postprocessing_labelmap(masks_labelmap, inverted_back, self.labels)

            if not self.terminate:
                self.results_ready.emit(labelmap)

        except Exception as e:
            print(f"Processing run failed: {e}")
        finally:
            self.finished.emit()


class AssessmentsWorker(WorkerParent):
    results_ready = Signal(str, str, str, str, str, str, dict)

    def __init__(self):
        super().__init__()
        self.mli_check_box_state = None
        self.asvd_check_box_state = None
        self.lines_spin_box_value = None
        self.min_length_spin_box_value = None
        self.scale_spin_box_value = None

    def set_mli_check_box_state(self, mli_check_box_state):
        self.mli_check_box_state = mli_check_box_state

    def set_asvd_check_box_state(self, asvd_check_box_state):
        self.asvd_check_box_state = asvd_check_box_state

    def set_lines_spin_box_value(self, lines_spin_box_value):
        self.lines_spin_box_value = lines_spin_box_value

    def set_min_length_spin_box_value(self, min_length_spin_box_value):
        self.min_length_spin_box_value = min_length_spin_box_value

    def set_scale_spin_box_value(self, scale_spin_box_value):
        self.scale_spin_box_value = scale_spin_box_value

    def run(self):
        mli = ""
        asvd = ""
        assessments_layer = None
        chords = ""
        stdev_chord_lengths = ""
        airspace_pixels = ""
        non_airspace_pixels = ""

        try:
            if not self.terminate:
                labelmap = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["POSTPROCESSING_LAYER"])

            if not self.terminate:
                if self.asvd_check_box_state:
                    asvd, airspace_pixels, non_airspace_pixels = calculate_airspace_volume_density(labelmap,
                                                                                                   self.labels)

            if not self.terminate:
                if self.mli_check_box_state:
                    mli, assessments_layer, chords, stdev_chord_lengths = calculate_mean_linear_intercept(
                        labelmap, self.lines_spin_box_value, self.min_length_spin_box_value,
                        self.scale_spin_box_value, self.labels
                    )

            if not self.terminate:
                self.results_ready.emit(str(asvd), str(mli), str(chords), str(stdev_chord_lengths),
                                        str(airspace_pixels), str(non_airspace_pixels),
                                        {"assessments_layer": assessments_layer})

        except Exception as e:
            print(f"Metrics calculation failed: {e}")
        finally:
            self.finished.emit()


class ExportWorker(WorkerParent):
    results_ready = Signal(dict, str)

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.selected_filter = None
        self.accumulated_results = None

    def set_file_path(self, file_path):
        self.file_path = file_path

    def set_selected_filter(self, selected_filter):
        self.selected_filter = selected_filter

    def set_accumulated_results(self, accumulated_results):
        self.accumulated_results = accumulated_results

    def run(self):
        data = None

        if not self.terminate:
            if self.selected_filter == "JSON Files (*.json)":
                data = export_operations.create_json_data(self.accumulated_results)
            elif self.selected_filter == "CSV Files (*.csv)":
                data = export_operations.create_csv_data(self.accumulated_results)

        if not self.terminate and data:
            with open(self.file_path, 'w') as file:
                file.write(data)

        self.finished.emit()
