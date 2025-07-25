import json
import pathlib
import traceback

import numpy as np
from qtpy.QtCore import QObject, Signal

from alveoleye.lungcv import model_operations
from alveoleye.lungcv.assessments import (
    calculate_airspace_volume_density,
    calculate_mean_linear_intercept,
)
from alveoleye.lungcv.postprocessor import (
    apply_dynamic_threshold,
    apply_manual_threshold,
    convert_to_grayscale,
    generate_postprocessing_labelmap,
    generate_processing_labelmap,
    invert_image_binary,
    remove_small_components,
)
import alveoleye._export_operations as export_operations
import alveoleye._layers_editor as layers_editor
from alveoleye._models import Result


class WorkerParent(QObject):
    finished = Signal()
    layers_config_data = None

    def __init__(self):
        super().__init__()
        self.napari_viewer = None
        self.layer_names = None
        self.labels = None
        self.callback = None
        self.terminate = False

        with open(pathlib.Path(__file__).resolve().parent / "config.json", 'r') as config_file:
            self.config_data = json.load(config_file)

    def set_napari_viewer(self, napari_viewer):
        self.napari_viewer = napari_viewer

    def set_layer_names(self, layer_names):
        self.layer_names = layer_names

    def set_labels(self, labels):
        self.labels = labels

    def set_callback(self, callback):
        self.callback = callback

    def cancel(self):
        self.terminate = True


class ProcessingWorker(WorkerParent):
    results_ready = Signal(dict, np.ndarray)

    def __init__(self):
        super().__init__()
        self.image_path = None
        self.image_shape = None
        self.use_ai = None
        self.weights = None
        self.confidence_threshold_value = None

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_use_ai(self, use_ai):
        self.use_ai = use_ai

    def set_image_shape(self, image_shape):
        self.image_shape = image_shape

    def set_weights(self, weights):
        self.weights = weights

    def set_confidence_threshold_value(self, confidence_threshold_value):
        self.confidence_threshold_value = confidence_threshold_value

    def run(self):
        try:
            if not self.terminate and (not self.use_ai or self.confidence_threshold_value == 100):
                inference_labelmap = np.zeros(self.image_shape[:2], dtype=np.uint8)
                model_output = {}

                self.results_ready.emit(model_output, inference_labelmap)
            else:
                if not self.terminate:
                    model = model_operations.init_trained_model(self.weights)

                if not self.terminate:
                    model_output = model_operations.run_prediction(self.image_path, model)

                if not self.terminate:
                    inference_labelmap = generate_processing_labelmap(model_output, self.image_shape,
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
        self.thresholding_check_box_value = None
        self.manual_threshold_value = None
        self.alveoli_minimum_size = None
        self.parenchyma_minimum_size = None
        self.confidence_threshold = None

    def set_thresholding_check_box_value(self, thresholding_check_box_value):
        self.thresholding_check_box_value = thresholding_check_box_value

    def set_manual_threshold_value(self, manual_threshold_value):
        self.manual_threshold_value = manual_threshold_value

    def set_alveoli_minimum_size(self, alveoli_minimum_size):
        self.alveoli_minimum_size = alveoli_minimum_size

    def set_parenchyma_minimum_size(self, parenchyma_minimum_size):
        self.parenchyma_minimum_size = parenchyma_minimum_size

    def threshold_according_to_method(self, image, callback):
        if self.thresholding_check_box_value:
            return apply_manual_threshold(image, self.manual_threshold_value, callback)

        return apply_dynamic_threshold(image, callback)

    def run(self):
        try:
            if not self.terminate:
                image = layers_editor.get_layers_by_names(self.napari_viewer, self.layer_names["INITIAL_LAYER"],
                                                          self.callback)

            if not self.terminate:
                grayscaled = convert_to_grayscale(image, self.callback)

            if not self.terminate:
                thresholded = self.threshold_according_to_method(grayscaled, self.callback)

            if not self.terminate:
                parenchyma_cleaned = remove_small_components(thresholded, self.parenchyma_minimum_size, self.callback)

            if not self.terminate:
                inverted = invert_image_binary(parenchyma_cleaned, self.callback)

            if not self.terminate:
                alveoli_cleaned = remove_small_components(inverted, self.alveoli_minimum_size, self.callback)

            if not self.terminate:
                inverted_back = invert_image_binary(alveoli_cleaned, self.callback)

            if not self.terminate:
                masks_labelmap = layers_editor.get_layers_by_names(self.napari_viewer,
                                                                   self.layer_names["PROCESSING_LAYER"], self.callback)

            if not self.terminate:
                labelmap = generate_postprocessing_labelmap(masks_labelmap, inverted_back, self.labels, self.callback)

            if not self.terminate:
                self.results_ready.emit(labelmap)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in post-processing: {e}")
            traceback.print_exc()
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
                labelmap = layers_editor.get_layers_by_names(self.napari_viewer, self.layer_names["POSTPROCESSING_LAYER"],
                                                             self.callback)

            if not self.terminate:
                if self.asvd_check_box_state:
                    asvd, airspace_pixels, non_airspace_pixels = calculate_airspace_volume_density(labelmap,
                                                                                                   self.labels)

            if not self.terminate:
                if self.mli_check_box_state:
                    mli, assessments_layer, chords, stdev_chord_lengths = calculate_mean_linear_intercept(
                        labelmap, self.lines_spin_box_value, self.min_length_spin_box_value,
                        self.scale_spin_box_value, self.labels, False, self.callback
                    )

            if not self.terminate:
                self.results_ready.emit(str(asvd), str(mli), str(chords), str(stdev_chord_lengths),
                                        str(airspace_pixels), str(non_airspace_pixels),
                                        {"assessments_layer": assessments_layer})

        except Exception as e:
            print(f"Error in metrics calculation: {e}")
        finally:
            self.finished.emit()


class ExportWorker(WorkerParent):
    results_ready = Signal(dict, str)

    def __init__(self):
        super().__init__()
        self.file_path = None # TODO: do we need?
        self.selected_filter = None  # TODO: do we need?
        self.parent_folder = ""
        self.project_name = ""
        self.metrics_ext = "csv"
        self.labelmap_ext = "tif"
        self.zip_it = False
        self.accumulated_results: list[Result] = []

    def set_parent_folder(self, f: str):
        self.parent_folder = f

    def set_project_name(self, n: str):
        self.project_name = n

    def set_metrics_format(self, e: str):
        self.metrics_ext = e

    def set_labelmap_format(self, e: str):
        self.labelmap_ext = e

    def set_zip(self, z: bool):
        self.zip_it = z
        
    def set_exp_rgb(self, r: bool):
        self.exp_rgb_colors = r

    def set_accumulated_results(self, res: list[Result]):
        self.accumulated_results = res

    def run(self):
        try:
            if self.terminate:
                return

            info = export_operations.export_results(
                results=self.accumulated_results,
                base_dir=self.parent_folder,
                project_name=self.project_name,
                metrics_format=self.metrics_ext,
                labelmap_ext=self.labelmap_ext,
                export_as_rgb=self.exp_rgb_colors,
                zip_it=self.zip_it,
            )

            self.results_ready.emit(info, "")
        except Exception as e:
            traceback.print_exc()
            self.results_ready.emit({}, str(e))
        finally:
            self.finished.emit()
