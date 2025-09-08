# boxes/process_box.py
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional, Tuple

import cv2
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QFileDialog

from .box_abstract import ActionBox, BoxState
import alveoleye._gui_creator as gui_creator
import alveoleye._layers_editor as layers_editor
from alveoleye._verifiers import verify_png_or_tiff
from alveoleye.boxes.postprocessing_box import PostprocessingActionBox


class ProcessingActionBox(ActionBox):
    model_output = None

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.image: Optional[object] = None

        self.import_image_line_edit = None
        self.import_weights_button_and_line_edit_layout = None
        self.use_ai_check_box = None
        self.import_weights_line_edit = None
        self.confidence_threshold_label_and_spin_box_layout = None
        self.confidence_threshold_spin_box = None

        self.box_id = 1

        self.create_ui_elements()
        self.create_ui_rules()
        self.set_default_weights()

        self._suppress_layer_event = False
        self.napari_viewer.layers.events.inserted.connect(self._on_layer_inserted)

    def build_worker(self):
        from alveoleye._workers import ProcessingWorker

        worker = ProcessingWorker()
        worker.set_napari_viewer(self.napari_viewer)
        worker.set_image_path(ActionBox.import_paths["image"])
        worker.set_use_ai(self.use_ai_check_box.isChecked())
        worker.set_weights(ActionBox.import_paths["weights"])
        worker.set_labels(self.labels_config_data)
        worker.set_image_shape(self.image.shape)
        worker.set_confidence_threshold_value(self.confidence_threshold_spin_box.value())
        return worker

    def set_default_weights(self) -> None:
        ActionBox.import_paths["weights"] = Path(__file__).resolve().parent.parent / "weights" / "default.pth"

    def create_ui_elements(self) -> None:
        import_image_btn = gui_creator.create_button_and_line_edit_layout(
            self.box_config_data["IMPORT_IMAGE_BUTTON_TEXT"],
            self.box_config_data["IMPORT_IMAGE_BUTTON_TOOLTIP_TEXT"],
            self.on_import_image_press,
            self.box_config_data["EMPTY_PATH_LINE_EDIT_TEXT"],
        )

        horizontal_line = gui_creator.create_horizontal_line_widget()

        use_ai_check_box = gui_creator.create_check_box_widget(
            self.box_config_data["USE_AI_CHECK_BOX_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["USE_AI_CHECK_BOX_TOOLTIP_TEXT"],
            self.box_config_data["USE_AI_CHECK_BOX_DEFAULT_VALUE"],
        )

        import_weights_btn = gui_creator.create_button_and_line_edit_layout(
            self.box_config_data["IMPORT_WEIGHTS_BUTTON_TEXT"],
            self.box_config_data["IMPORT_WEIGHTS_BUTTON_TOOLTIP_TEXT"],
            self.on_import_weights_press,
            self.box_config_data["DEFAULT_WEIGHTS_NAME"],
        )

        conf_spin = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["CONFIDENCE_THRESHOLD_LABEL_TEXT"],
            self.box_config_data["MINIMUM_CONFIDENCE_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_STEP"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_SUFFIX"],
        )

        self.import_image_line_edit = import_image_btn[2]
        self.import_weights_button_and_line_edit_layout = import_weights_btn[0]
        self.use_ai_check_box = use_ai_check_box
        self.import_weights_line_edit = import_weights_btn[2]
        self.confidence_threshold_label_and_spin_box_layout = conf_spin[0]
        self.confidence_threshold_spin_box = conf_spin[2]

        ui_elements = [
            import_image_btn[0],
            horizontal_line,
            use_ai_check_box,
            self.import_weights_button_and_line_edit_layout,
            self.confidence_threshold_label_and_spin_box_layout,
        ]

        self.create_action_box_layout(
            ui_elements,
            self.box_config_data["ACTION_BUTTON_TEXT"],
            self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"],
        )

    def create_ui_rules(self) -> None:
        from .box_abstract import ActionBox, BoxState  # for clarity in lambdas

        self.rules_engine.toggle_visibility_based_on_condition(lambda: ActionBox.import_paths["image"] is not None, self.action_button)
        self.rules_engine.toggle_visibility_based_on_condition(lambda: ActionBox.import_paths["image"] is not None, self.import_image_line_edit)
        self.rules_engine.toggle_visibility_based_on_checkbox_state(self.use_ai_check_box, [self.import_weights_button_and_line_edit_layout, self.confidence_threshold_label_and_spin_box_layout])

        super().create_ui_rules()

    def open_file_dialogue(self, title: str, accepted_extensions: str) -> Tuple[Optional[str], Optional[str]]:
        key = "weights" if "weights" in title.lower() else "image"
        parent_dir = str(Path.home()) if ActionBox.import_paths[key] is None else str(Path(ActionBox.import_paths[key]).parent)
        file_path = QFileDialog.getOpenFileName(self, title, parent_dir, accepted_extensions)[0]
        return (file_path, Path(file_path).name) if file_path else (None, None)

    def on_import_press(self, file_type: str, file_line_edit, dialogue_text: str, accepted_file_formats: str) -> bool:
        file_path, file_name = self.open_file_dialogue(dialogue_text, accepted_file_formats)
        if not file_path:
            return False

        if file_type == "image":
            ok, *_ = verify_png_or_tiff(file_path)
            if not ok:
                return False

        if self.state is BoxState.RUNNING:
            self.cancel_action()

        ActionBox.import_paths[file_type] = file_path
        file_line_edit.setText(file_name)
        self.rules_engine.evaluate_rules()
        return True

    def on_import_image_press(self) -> None:
        if not self.on_import_press(
            "image",
            self.import_image_line_edit,
            self.box_config_data["IMAGE_FILE_DIALOGUE_TEXT"],
            self.box_config_data["IMAGE_ACCEPTED_FILE_FORMATS"],
        ):
            return
        self._handle_new_image_from_path(ActionBox.import_paths["image"])

    def on_import_weights_press(self) -> None:
        self.on_import_press(
            "weights",
            self.import_weights_line_edit,
            self.box_config_data["WEIGHTS_FILE_DIALOGUE_TEXT"],
            self.box_config_data["WEIGHTS_ACCEPTED_FILE_FORMATS"],
        )

    def _remove_layer(self, layer) -> None:
        self._suppress_layer_event = True
        try:
            if layer in self.napari_viewer.layers:
                self.napari_viewer.layers.remove(layer)
        finally:
            self._suppress_layer_event = False

    def _consume_and_load(self, path: str, layer) -> None:
        self._remove_layer(layer)
        self._handle_new_image_from_path(path)

    def _on_layer_inserted(self, event) -> None:
        if self._suppress_layer_event:
            return

        layer = getattr(event, "value", None)
        if layer is None or not hasattr(layer, "data"):
            return

        src = getattr(layer, "source", None)
        path = getattr(src, "path", None) if src is not None else None

        if isinstance(path, (list, tuple)) and path:
            path = path[0]
        if not path or not Path(path).exists():
            return

        ok, *_ = verify_png_or_tiff(path)
        if not ok:
            QTimer.singleShot(0, functools.partial(self._remove_layer, layer))
            return

        QTimer.singleShot(0, functools.partial(self._consume_and_load, path, layer))

    def _handle_new_image_from_path(self, path: str) -> None:
        try:
            ok, *_ = verify_png_or_tiff(path)
            if not ok:
                return

            if self.state is BoxState.RUNNING:
                self.cancel_action()

            pstr = str(path)
            ActionBox.import_paths["image"] = pstr
            self.import_image_line_edit.setText(Path(pstr).name)
            self.rules_engine.evaluate_rules()

            self.image = cv2.imread(pstr)
            if self.image is None:
                raise RuntimeError(f"cv2.imread failed for: {pstr}")

            self._suppress_layer_event = True
            try:
                layers_editor.remove_all_layers(self.napari_viewer)
                layers_editor.update_layers(
                    self.napari_viewer,
                    self.layers_config_data["INITIAL_LAYER"],
                    self.image,
                    self.colormap_config_data,
                    self.labels_config_data,
                    False,
                    False,
                )
            finally:
                self._suppress_layer_event = False

            # ensure pristine state (duplicated sequence preserved for behavioral parity)
            layers_editor.remove_all_layers(self.napari_viewer)
            layers_editor.update_layers(
                self.napari_viewer,
                self.layers_config_data["INITIAL_LAYER"],
                self.image,
                self.colormap_config_data,
                self.labels_config_data,
                False,
                False,
            )

            self._clear_results_after_new_image()
            self.set_image_threshold_value()
            self.broadcast_cancel_message()
            self.broadcast_update_step(0)

        except Exception as e:
            print(f"[-] Failed preparing image: {e}")
            self.broadcast_cancel_message()
            self.broadcast_update_step(0)

    def _clear_results_after_new_image(self) -> None:
        ActionBox.current_results = []
        self.evaluate_box_rules()

    def set_image_threshold_value(self) -> None:
        image = layers_editor.get_layers_by_names(self.napari_viewer, self.layers_config_data["INITIAL_LAYER"])
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu_value = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_OTSU)[0] + 20
        PostprocessingActionBox.threshold_value = round(otsu_value)

    def on_results_ready(self, model_output, inference_labelmap) -> None:
        ProcessingActionBox.model_output = model_output

        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["ASSESSMENTS_LAYER"])
        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["POSTPROCESSING_LAYER"])
        layers_editor.update_layers(
            self.napari_viewer,
            self.layers_config_data["PROCESSING_LAYER"],
            inference_labelmap,
            self.colormap_config_data,
            self.labels_config_data,
            True,
            True,
        )

        ActionBox.current_results = []
        super().on_results_ready()

        ActionBox.current_use_computer_vision = self.use_ai_check_box.isChecked()
        ActionBox.current_min_confidence = self.confidence_threshold_spin_box.value()
