import json
from pathlib import Path
from typing import Any
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
    finished: Signal = Signal()
    layers_config_data: dict | None = None

    def __init__(self) -> None:
        super().__init__()
        self.napari_viewer: Any = None
        self.layer_names: dict[str, str] | None = None
        self.labels: dict[str, Any] | None = None
        self.terminate: bool = False
        config_path: Path = Path(__file__).resolve().parent / "config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
            self.config_data: dict[str, Any] = json.load(config_file)

    def set_napari_viewer(self, napari_viewer: Any) -> None:
        self.napari_viewer = napari_viewer

    def set_layer_names(self, layer_names: dict[str, str]) -> None:
        self.layer_names = layer_names

    def set_labels(self, labels: dict[str, Any]) -> None:
        self.labels = labels

    def cancel(self) -> None:
        self.terminate = True

@typechecked
class ProcessingWorker(WorkerParent):
    # Signals in PyQt do not have static typing for their arguments, kept as original
    results_ready: Signal = Signal(dict, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self.image_path: Path | None = None
        self.image_shape: tuple[int, ...] | None = None
        self.weights: Path | None = None
        self.confidence_threshold_value: int | None = None

    def set_image_path(self, image_path: Path | str) -> None:
        self.image_path = Path(image_path)

    def set_image_shape(self, image_shape: tuple[int, ...]) -> None:
        self.image_shape = image_shape

    def set_weights(self, weights: Path | str) -> None:
        self.weights = Path(weights)

    def set_confidence_threshold_value(self, confidence_threshold_value: int) -> None:
        self.confidence_threshold_value = confidence_threshold_value

    def run(self) -> None:
        try:
            if self.terminate:
                return

            assert self.weights is not None, "Weights must be set"
            model = model_operations.init_trained_model(self.weights)

            if self.terminate:
                return

            assert self.image_path is not None, "Image path must be set"
            model_output = model_operations.run_prediction(self.image_path, model)

            if self.terminate:
                return

            assert self.image_shape is not None, "Image shape must be set"
            assert self.confidence_threshold_value is not None, "Confidence threshold must be set"
            assert self.labels is not None, "Labels must be set"
            inference_labelmap = generate_processing_labelmap(
                model_output, self.image_shape, self.confidence_threshold_value, self.labels
            )

            if self.terminate:
                return

            self.results_ready.emit(model_output, inference_labelmap)
        except Exception as e:
            print(f"Error in processing: {e}")
        finally:
            self.finished.emit()

@typechecked
class PostprocessingWorker(WorkerParent):
    results_ready: Signal = Signal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        self.model_output: dict | None = None
        self.thresholding_check_box_value: bool | None = None
        self.manual_threshold_value: int | None = None
        self.alveoli_minimum_size: int | None = None
        self.parenchyma_minimum_size: int | None = None

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
            assert self.manual_threshold_value is not None, "Manual threshold value must be set"
            return apply_manual_threshold(image, self.manual_threshold_value)
        return apply_dynamic_threshold(image)

    def run(self) -> None:
        try:
            if self.terminate:
                return

            assert self.napari_viewer is not None, "Napari viewer must be set"
            assert self.layer_names is not None, "Layer names must be set"
            image = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["INITIAL_LAYER"])

            if self.terminate:
                return

            grayscaled = convert_to_grayscale(image)

            if self.terminate:
                return

            thresholded = self.threshold_according_to_method(grayscaled)

            if self.terminate:
                return

            assert self.parenchyma_minimum_size is not None, "Parenchyma minimum size must be set"
            parenchyma_cleaned = remove_small_components(thresholded, self.parenchyma_minimum_size)

            if self.terminate:
                return

            inverted = invert_image_binary(parenchyma_cleaned)

            if self.terminate:
                return

            assert self.alveoli_minimum_size is not None, "Alveoli minimum size must be set"
            alveoli_cleaned = remove_small_components(inverted, self.alveoli_minimum_size)

            if self.terminate:
                return

            inverted_back = invert_image_binary(alveoli_cleaned)

            if self.terminate:
                return

            masks_labelmap = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["PROCESSING_LAYER"])

            if self.terminate:
                return

            assert self.labels is not None, "Labels must be set"
            labelmap = generate_postprocessing_labelmap(masks_labelmap, inverted_back, self.labels)

            if self.terminate:
                return

            self.results_ready.emit(labelmap)
        except Exception as e:
            print(f"Error in post-processing: {e}")
        finally:
            self.finished.emit()

@typechecked
class AssessmentsWorker(WorkerParent):
    results_ready: Signal = Signal(str, str, str, str, str, str, dict)

    def __init__(self) -> None:
        super().__init__()
        self.mli_check_box_state: bool | None = None
        self.asvd_check_box_state: bool | None = None
        self.lines_spin_box_value: int | None = None
        self.min_length_spin_box_value: int | None = None
        self.scale_spin_box_value: float | None = None

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
        assessments_layer: Any = None
        try:
            if self.terminate:
                return

            assert self.napari_viewer is not None, "Napari viewer must be set"
            assert self.layer_names is not None, "Layer names must be set"
            labelmap = layers_editor.get_layer_by_name(self.napari_viewer, self.layer_names["POSTPROCESSING_LAYER"])

            if self.terminate:
                return

            if self.asvd_check_box_state:
                assert labelmap is not None and self.labels is not None
                asvd, airspace_pixels, non_airspace_pixels = calculate_airspace_volume_density(labelmap, self.labels)

            if self.terminate:
                return

            if self.mli_check_box_state:
                assert labelmap is not None and self.lines_spin_box_value is not None and self.min_length_spin_box_value is not None and self.scale_spin_box_value is not None and self.labels is not None
                mli, assessments_layer, chords, stdev_chord_lengths = calculate_mean_linear_intercept(
                    labelmap,
                    self.lines_spin_box_value,
                    self.min_length_spin_box_value,
                    self.scale_spin_box_value,
                    self.labels
                )

            if self.terminate:
                return

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
    results_ready: Signal = Signal(dict, str)

    def __init__(self) -> None:
        super().__init__()
        self.file_path: Path | None = None
        self.selected_filter: str | None = None
        self.accumulated_results: list[tuple] | None = None

    def set_file_path(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)

    def set_selected_filter(self, selected_filter: str) -> None:
        self.selected_filter = selected_filter

    def set_accumulated_results(self, results: list[tuple]) -> None:
        self.accumulated_results = results

    def run(self) -> None:
        if self.terminate:
            self.finished.emit()
            return

        data: str | None = None
        if self.selected_filter == "JSON Files (*.json)":
            assert self.accumulated_results is not None
            data = export_operations.create_json_data(self.accumulated_results)
        elif self.selected_filter == "CSV Files (*.csv)":
            assert self.accumulated_results is not None
            data = export_operations.create_csv_data(self.accumulated_results)

        if not self.terminate and data is not None:
            assert self.file_path is not None, "File path must be set"
            with self.file_path.open("w", encoding="utf-8") as file:
                file.write(data)

        self.finished.emit()