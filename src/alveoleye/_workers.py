import json
import pathlib
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

import numpy as np
from qtpy.QtCore import QObject, Signal
from typeguard import typechecked

from alveoleye.lungcv import model_operations
from alveoleye.lungcv.postprocessor import (
    apply_manual_threshold, apply_dynamic_threshold,
    generate_postprocessing_labelmap, remove_small_components,
    convert_to_grayscale, invert_image_binary, generate_processing_labelmap
)
from alveoleye.lungcv.assessments import (
    calculate_mean_linear_intercept, calculate_airspace_volume_density
)
import alveoleye._layers_editor as layers_editor
import alveoleye._export_operations as export_operations


@typechecked
class WorkerParent(QObject):
    finished = Signal()
    layers_config_data: Optional[dict] = None

    def __init__(self) -> None:
        super().__init__()
        self.napari_viewer: Any = None
        self.layer_names: Optional[dict] = None
        self.labels: Optional[dict] = None
        self.terminate: bool = False

        with open(pathlib.Path(__file__).resolve().parent / "config.json", 'r') as config_file:
            self.config_data: dict = json.load(config_file)

    def set_napari_viewer(self, napari_viewer: Any) -> None:
        self.napari_viewer = napari_viewer

    def set_layer_names(self, layer_names: dict) -> None:
        self.layer_names = layer_names

    def set_labels(self, labels: dict) -> None:
        self.labels = labels

    def cancel(self) -> None:
        self.terminate = True


@typechecked
class ProcessingWorker(WorkerParent):
    results_ready = Signal(dict, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self.image_path: Optional[str] = None
        self.image_shape: Optional[tuple] = None
        self.weights: Optional[str] = None
        self.confidence_threshold_value: Optional[int] = None

    def set_image_path(self, image_path: str) -> None:
        self.image_path = image_path

    def set_image_shape(self, image_shape: tuple) -> None:
        self.image_shape = image_shape

    def set_weights(self, weights: Union[str, Path]) -> None:
        self.weights = weights

    def set_confidence_threshold_value(self, confidence_threshold_value: int) -> None:
        self.confidence_threshold_value = confidence_threshold_value

    def run(self) -> None:
        try:
            if not self.terminate:
                model = model_operations.init_trained_model(self.weights)

            if not self.terminate:
                model_output = model_operations.run_prediction(self.image_path, model)

            if not self.terminate:
                inference_labelmap = generate_processing_labelmap(
                    model_output, self.image_shape, self.confidence_threshold_value, self.labels
                )

            if not self.terminate:
                self.results_ready.emit(model_output, inference_labelmap)

        except Exception as e:
            print(f"Error in processing: {e}")
        finally:
            self.finished.emit()


@typechecked
class PostprocessingWorker(WorkerParent):
    results_ready = Signal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self.model_output: Optional[dict] = None
        self.thresholding_check_box_value: Optional[bool] = None
        self.manual_threshold_value: Optional[int] = None
        self.alveoli_minimum_size: Optional[int] = None
        self.parenchyma_minimum_size: Optional[int] = None

    def set_model_output(self, model_output: dict) -> None:
        self.model_output = model_output

    def set_thresholding_check_box_value(self, value: bool) -> None:
        self.thresholding_check_box_value = value

    def set_manual_threshold_value(self, value: int) -> None:
        self.manual_threshold_value = value

    def set_alveoli_minimum_size(self, value: int) -> None:
        self.alveoli_minimum_size = value

    def set_parenchyma_minimum_size(self, value: int) -> None:
        self.parenchyma_minimum_size = value

    def threshold_according_to_method(self, image: np.ndarray) -> np.ndarray:
        if self.thresholding_check_box_value:
            return apply_manual_threshold(image, self.manual_threshold_value)
        return apply_dynamic_threshold(image)

    def run(self) -> None:
        try:
            if not self.terminate:
                image = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["INITIAL_LAYER"])

            if not self.terminate:
                grayscaled = convert_to_grayscale(image)

            if not self.terminate:
                thresholded = self.threshold_according_to_method(grayscaled)

            if not self.terminate:
                parenchyma_cleaned = remove_small_components(thresholded, self.parenchyma_minimum_size)

            if not self.terminate:
                inverted = invert_image_binary(parenchyma_cleaned)

            if not self.terminate:
                alveoli_cleaned = remove_small_components(inverted, self.alveoli_minimum_size)

            if not self.terminate:
                inverted_back = invert_image_binary(alveoli_cleaned)

            if not self.terminate:
                masks_labelmap = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["PROCESSING_LAYER"])

            if not self.terminate:
                labelmap = generate_postprocessing_labelmap(masks_labelmap, inverted_back, self.labels)

            if not self.terminate:
                self.results_ready.emit(labelmap)

        except Exception as e:
            print(f"Error in post-processing: {e}")
        finally:
            self.finished.emit()


@typechecked
class AssessmentsWorker(WorkerParent):
    results_ready = Signal(str, str, str, str, str, str, dict)

    def __init__(self) -> None:
        super().__init__()
        self.mli_check_box_state: Optional[bool] = None
        self.asvd_check_box_state: Optional[bool] = None
        self.lines_spin_box_value: Optional[int] = None
        self.min_length_spin_box_value: Optional[int] = None
        self.scale_spin_box_value: Optional[float] = None

    def set_mli_check_box_state(self, value: bool) -> None:
        self.mli_check_box_state = value

    def set_asvd_check_box_state(self, value: bool) -> None:
        self.asvd_check_box_state = value

    def set_lines_spin_box_value(self, value: int) -> None:
        self.lines_spin_box_value = value

    def set_min_length_spin_box_value(self, value: int) -> None:
        self.min_length_spin_box_value = value

    def set_scale_spin_box_value(self, value: float) -> None:
        self.scale_spin_box_value = value

    def run(self) -> None:
        mli = ""
        asvd = ""
        chords = ""
        stdev_chord_lengths = ""
        airspace_pixels = ""
        non_airspace_pixels = ""
        assessments_layer = None

        try:
            if not self.terminate:
                labelmap = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["POSTPROCESSING_LAYER"])

            if not self.terminate and self.asvd_check_box_state:
                asvd, airspace_pixels, non_airspace_pixels = calculate_airspace_volume_density(labelmap, self.labels)

            if not self.terminate and self.mli_check_box_state:
                mli, assessments_layer, chords, stdev_chord_lengths = calculate_mean_linear_intercept(
                    labelmap,
                    self.lines_spin_box_value,
                    self.min_length_spin_box_value,
                    self.scale_spin_box_value,
                    self.labels
                )

            if not self.terminate:
                self.results_ready.emit(
                    str(asvd), str(mli), str(chords), str(stdev_chord_lengths),
                    str(airspace_pixels), str(non_airspace_pixels),
                    {"assessments_layer": assessments_layer}
                )

        except Exception as e:
            print(f"Error in metrics calculation: {e}")
        finally:
            self.finished.emit()


@typechecked
class ExportWorker(WorkerParent):
    results_ready = Signal(dict, str)

    def __init__(self) -> None:
        super().__init__()
        self.file_path: Optional[str] = None
        self.selected_filter: Optional[str] = None
        self.accumulated_results: Optional[List[tuple]] = None

    def set_file_path(self, file_path: str) -> None:
        self.file_path = file_path

    def set_selected_filter(self, selected_filter: str) -> None:
        self.selected_filter = selected_filter

    def set_accumulated_results(self, results: List[tuple]) -> None:
        self.accumulated_results = results

    def run(self) -> None:
        data: Optional[str] = None

        if not self.terminate:
            if self.selected_filter == "JSON Files (*.json)":
                data = export_operations.create_json_data(self.accumulated_results)
            elif self.selected_filter == "CSV Files (*.csv)":
                data = export_operations.create_csv_data(self.accumulated_results)

        if not self.terminate and data:
            with open(self.file_path, 'w') as file:
                file.write(data)

        self.finished.emit()
