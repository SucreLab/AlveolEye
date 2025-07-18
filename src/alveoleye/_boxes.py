import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from qtpy.QtWidgets import QFileDialog

from alveoleye._action_box import ActionBox
import alveoleye._gui_creator as gui_creator
import alveoleye._layers_editor as layers_editor
from alveoleye._export_operations import is_real_writable_dir
from alveoleye._models import Result
from alveoleye._workers import (
    AssessmentsWorker,
    ExportWorker,
    PostprocessingWorker,
    ProcessingWorker,
)


class ProcessingActionBox(ActionBox):
    model_output = None

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.image = None

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

    def set_default_weights(self):
        ActionBox.import_paths["weights"] = Path(__file__).resolve().parent.parent / "weights" / "default.pth"

    def thread_worker(self):
        self.worker = ProcessingWorker()

        self.worker.set_napari_viewer(self.napari_viewer)
        self.worker.set_image_path(ActionBox.import_paths["image"])
        self.worker.set_use_ai(self.use_ai_check_box.isChecked())
        self.worker.set_weights(ActionBox.import_paths["weights"])
        self.worker.set_labels(self.labels_config_data)
        self.worker.set_image_shape(self.image.shape)
        self.worker.set_confidence_threshold_value(self.confidence_threshold_spin_box.value())

        super().thread_worker()

    def create_ui_elements(self):
        import_image_button_and_line_edit = gui_creator.create_button_and_line_edit_layout(
            self.box_config_data["IMPORT_IMAGE_BUTTON_TEXT"],
            self.box_config_data["IMPORT_IMAGE_BUTTON_TOOLTIP_TEXT"],
            self.on_import_image_press,
            self.box_config_data["EMPTY_PATH_LINE_EDIT_TEXT"]
        )

        horizontal_line = gui_creator.create_horizontal_line_widget()

        use_ai_check_box = gui_creator.create_check_box_widget(
            self.box_config_data["USE_AI_CHECK_BOX_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["USE_AI_CHECK_BOX_TOOLTIP_TEXT"],
            self.box_config_data["USE_AI_CHECK_BOX_DEFAULT_VALUE"]
        )

        import_weights_button_and_line_edit = gui_creator.create_button_and_line_edit_layout(
            self.box_config_data["IMPORT_WEIGHTS_BUTTON_TEXT"],
            self.box_config_data["IMPORT_WEIGHTS_BUTTON_TOOLTIP_TEXT"],
            self.on_import_weights_press,
            self.box_config_data["DEFAULT_WEIGHTS_NAME"]
        )
        confidence_threshold_label_and_spin_box = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["CONFIDENCE_THRESHOLD_LABEL_TEXT"],
            self.box_config_data["MINIMUM_CONFIDENCE_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_STEP"],
            self.box_config_data["CONFIDENCE_THRESHOLD_SPIN_BOX_SUFFIX"]
        )

        import_image_button_and_line_edit_layout = import_image_button_and_line_edit[0]
        import_weights_button_and_line_edit_layout = import_weights_button_and_line_edit[0]
        confidence_threshold_label_and_spin_box_layout = confidence_threshold_label_and_spin_box[0]

        self.import_image_line_edit = import_image_button_and_line_edit[2]
        self.import_weights_button_and_line_edit_layout = import_weights_button_and_line_edit_layout
        self.use_ai_check_box = use_ai_check_box
        self.import_weights_line_edit = import_weights_button_and_line_edit[2]
        self.confidence_threshold_spin_box = confidence_threshold_label_and_spin_box[2]
        self.confidence_threshold_label_and_spin_box_layout = confidence_threshold_label_and_spin_box_layout

        ui_elements = [import_image_button_and_line_edit_layout,
                       horizontal_line,
                       use_ai_check_box,
                       import_weights_button_and_line_edit_layout,
                       confidence_threshold_label_and_spin_box_layout]

        self.create_action_box_layout(ui_elements,
                                      self.box_config_data["ACTION_BUTTON_TEXT"],
                                      self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"])

    def create_ui_rules(self):
        self.rules_engine.add_rule([lambda: ActionBox.import_paths["image"] is None,
                                    lambda: ActionBox.import_paths["weights"] is None,
                                    lambda: not self.state == 2],
                                   lambda: gui_creator.toggle(False, self.action_button))

        self.rules_engine.add_rule(lambda: ActionBox.import_paths["image"] is not None,
                                   lambda: gui_creator.toggle(True, self.action_button))

        self.rules_engine.add_rule(lambda: ActionBox.import_paths["image"] is None,
                                   lambda: gui_creator.toggle(False, self.import_image_line_edit))
        self.rules_engine.add_rule(lambda: ActionBox.import_paths["image"] is not None,
                                   lambda: gui_creator.toggle(True, self.import_image_line_edit))

        self.rules_engine.add_rule(lambda: self.use_ai_check_box.isChecked(),
                                   lambda: gui_creator.toggle(True, [self.import_weights_button_and_line_edit_layout,
                                                                                self.confidence_threshold_label_and_spin_box_layout]))
        self.rules_engine.add_rule(lambda: not self.use_ai_check_box.isChecked(),
                                   lambda: gui_creator.toggle(False, [self.import_weights_button_and_line_edit_layout,
                                                                                 self.confidence_threshold_label_and_spin_box_layout]))

        super().create_ui_rules()

    def open_file_dialogue(self, title, accepted_extensions, parent_directory):
        parent_directory = str(Path(__file__).resolve().parent.parent / parent_directory)
        file_path = QFileDialog.getOpenFileName(self, title, parent_directory, accepted_extensions)[0]

        return file_path, Path(file_path).name if file_path else (None, None)

    def on_import_press(self, file_type, file_line_edit, dialogue_text, accepted_file_formats, parent_directory):
        file_path, file_name = self.open_file_dialogue(dialogue_text, accepted_file_formats, parent_directory)

        if not file_path:
            return False

        if self.state == 1:
            self.cancel_action()

        ActionBox.import_paths[file_type] = file_path
        file_line_edit.setText(file_name)
        self.rules_engine.evaluate_rules()
        return True

    def on_import_image_press(self):
        if not self.on_import_press("image", self.import_image_line_edit,
                                    self.box_config_data["IMAGE_FILE_DIALOGUE_TEXT"],
                                    self.box_config_data["IMAGE_ACCEPTED_FILE_FORMATS"],
                                    self.box_config_data["IMAGES_FOLDER_PATH"]):
            return
        try:
            self.image = cv2.imread(ActionBox.import_paths["image"])
        except Exception as e:
            print(f"[-] Failed reading image {e}")
            self.broadcast_cancel_message()
            self.broadcast_step_change_message(0)
            return

        layers_editor.remove_all_layers(self.napari_viewer)
        layers_editor.update_layers(self.napari_viewer, self.layers_config_data["INITIAL_LAYER"], self.image,
                                    self.colormap_config_data, self.labels_config_data, False, False)

        self.set_image_threshold_value()
        self.broadcast_cancel_message()
        self.broadcast_step_change_message(0)

    def on_import_weights_press(self):
        self.on_import_press("weights", self.import_weights_line_edit,
                             self.box_config_data["WEIGHTS_FILE_DIALOGUE_TEXT"],
                             self.box_config_data["WEIGHTS_ACCEPTED_FILE_FORMATS"],
                             self.box_config_data["WEIGHTS_FOLDER_PATH"])

    def set_image_threshold_value(self):
        image = layers_editor.get_layers_by_names(self.napari_viewer, self.layers_config_data["INITIAL_LAYER"])
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu_value = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_OTSU)[0] + 20
        threshold_value = round(otsu_value)
        PostprocessingActionBox.threshold_value = threshold_value

    def on_results_ready(self, model_output, inference_labelmap):
        ProcessingActionBox.model_output = model_output

        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["ASSESSMENTS_LAYER"])
        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["POSTPROCESSING_LAYER"])
        layers_editor.update_layers(self.napari_viewer, self.layers_config_data["PROCESSING_LAYER"],
                                    inference_labelmap, self.colormap_config_data, self.labels_config_data, True, True)

        super().on_results_ready()

        ActionBox.current_use_computer_vision = self.use_ai_check_box.isChecked()
        ActionBox.current_min_confidence = self.confidence_threshold_spin_box.value()


class PostprocessingActionBox(ActionBox):
    threshold_value = None

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.thresholding_check_box = None
        self.thresholding_spin_box = None

        self.clean_alveoli_spin_box = None
        self.clean_parenchyma_spin_box = None

        self.box_id = 2

        self.create_ui_elements()
        self.create_ui_rules()

    def thread_worker(self):
        self.worker = PostprocessingWorker()
        self.worker.set_napari_viewer(self.napari_viewer)
        self.worker.set_layer_names(self.layers_config_data)
        self.worker.set_labels(self.labels_config_data)
        self.worker.set_thresholding_check_box_value(self.thresholding_check_box.isChecked())
        self.worker.set_manual_threshold_value(self.thresholding_spin_box.value())
        self.worker.set_alveoli_minimum_size(self.clean_alveoli_spin_box.value())
        self.worker.set_parenchyma_minimum_size(self.clean_parenchyma_spin_box.value())

        super().thread_worker()

    def create_ui_elements(self):
        thresholding_check_box_and_spin_box = gui_creator.create_check_box_and_spin_box_layout(
            self.box_config_data["THRESHOLDING_BOX_TEXT"],
            self.box_config_data["MANUAL_THRESHOLD_CHECKBOX_TOOLTIP_TEXT"],
            self.box_config_data["MANUAL_THRESHOLD_SPIN_BOX_TOOLTIP_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["THRESHOLDING_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["THRESHOLDING_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["THRESHOLDING_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["THRESHOLDING_SPIN_BOX_STEP"]
        )
        clean_alveoli_label_and_spin_box = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["CLEAN_ALVEOLI_LABEL_TEXT"],
            self.box_config_data["REMOVE_SMALL_PARTICLES_TOOLTIP_TEXT"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_STEP"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_SUFFIX"]
        )
        clean_parenchyma_label_and_spin_box = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["CLEAN_PARENCHYMA_LABEL_TEXT"],
            self.box_config_data["REMOVE_SMALL_HOLES_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_STEP"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_SUFFIX"]
        )

        thresholding_check_box_and_spin_box_layout = thresholding_check_box_and_spin_box[0]
        clean_alveoli_label_and_spin_box_layout = clean_alveoli_label_and_spin_box[0]
        clean_parenchyma_label_and_spin_box_layout = clean_parenchyma_label_and_spin_box[0]

        self.thresholding_check_box = thresholding_check_box_and_spin_box[1]
        self.thresholding_spin_box = thresholding_check_box_and_spin_box[2]
        self.clean_alveoli_spin_box = clean_alveoli_label_and_spin_box[2]
        self.clean_parenchyma_spin_box = clean_parenchyma_label_and_spin_box[2]

        ui_elements = [thresholding_check_box_and_spin_box_layout,
                       clean_alveoli_label_and_spin_box_layout,
                       clean_parenchyma_label_and_spin_box_layout]

        self.create_action_box_layout(ui_elements,
                                      self.box_config_data["ACTION_BUTTON_TEXT"],
                                      self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"])

    def create_ui_rules(self):
        self.rules_engine.add_rule([lambda: PostprocessingActionBox.threshold_value != None,
                                    lambda: ActionBox.step == 0],
                                   lambda: self.thresholding_spin_box.setValue(PostprocessingActionBox.threshold_value))
        self.rules_engine.add_rule(lambda: self.thresholding_check_box.isChecked(),
                                   lambda: gui_creator.toggle(True, self.thresholding_spin_box))
        self.rules_engine.add_rule(lambda: not self.thresholding_check_box.isChecked(),
                                   lambda: gui_creator.toggle(False, self.thresholding_spin_box))

        self.rules_engine.add_rule(lambda: ActionBox.step == 0,
                                   lambda: gui_creator.toggle(False, self.action_button))

        self.rules_engine.add_rule([lambda: self.state == 0, lambda: ActionBox.step == 1],
                                   lambda: gui_creator.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: self.state == 0, lambda: ActionBox.step == 2],
                                   lambda: gui_creator.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: self.state == 0, lambda: ActionBox.step == 3],
                                   lambda: gui_creator.toggle(True, self.action_button))

        super().create_ui_rules()

    def on_results_ready(self, labelmap):
        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["ASSESSMENTS_LAYER"])
        layers_editor.update_layers(self.napari_viewer, self.layers_config_data["POSTPROCESSING_LAYER"], labelmap,
                                    self.colormap_config_data, self.labels_config_data, True, True)

        super().on_results_ready()

        # store post‐processing arguments for export
        ActionBox.current_used_manual_threshold = self.thresholding_check_box.isChecked()

        if ActionBox.current_used_manual_threshold:
            ActionBox.current_threshold_value = self.thresholding_spin_box.value()
        else:
            ActionBox.current_threshold_value = PostprocessingActionBox.threshold_value

        ActionBox.current_remove_small_particles = self.clean_alveoli_spin_box.value()
        ActionBox.current_remove_small_holes = self.clean_parenchyma_spin_box.value()


class AssessmentsActionBox(ActionBox):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.lines_spin_box = None
        self.min_length_spin_box = None
        self.scale_spin_box = None

        self.lines_label_and_spin_box_layout = None
        self.min_length_label_and_spin_box_layout = None
        self.scale_label_and_spin_box_layout = None

        self.mli_line_edit = None
        self.mli_check_box = None
        self.mli_check_box_and_line_edit_layout = None

        self.asvd_line_edit = None
        self.asvd_check_box = None
        self.asvd_check_box_and_line_edit_layout = None

        self.box_id = 3

        self.create_ui_elements()
        self.create_ui_rules()

    def thread_worker(self):
        self.worker = AssessmentsWorker()

        self.worker.set_napari_viewer(self.napari_viewer)
        self.worker.set_layer_names(self.layers_config_data)
        self.worker.set_labels(self.labels_config_data)
        self.worker.set_asvd_check_box_state(self.asvd_check_box.isChecked())
        self.worker.set_mli_check_box_state(self.mli_check_box.isChecked())
        self.worker.set_lines_spin_box_value(self.lines_spin_box.value())
        self.worker.set_min_length_spin_box_value(self.min_length_spin_box.value())
        self.worker.set_scale_spin_box_value(round(self.scale_spin_box.value(), 5))

        super().thread_worker()

    def create_ui_elements(self):
        (self.asvd_check_box_and_line_edit_layout,
         self.asvd_check_box,
         self.asvd_line_edit) = gui_creator.create_check_box_and_line_edit_layout(
            self.box_config_data["ASVD_CHECK_BOX_TITLE"],
            self.box_config_data["ASVD_CHECKBOX_TOOLTIP_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["ASVD_RESULT_LINE_EDIT_DEFAULT"])

        horizontal_line_one = gui_creator.create_horizontal_line_widget()

        (self.mli_check_box_and_line_edit_layout,
         self.mli_check_box,
         self.mli_line_edit) = gui_creator.create_check_box_and_line_edit_layout(
            self.box_config_data["MLI_CHECK_BOX_TITLE"],
            self.box_config_data["MLI_CHECKBOX_TOOLTIP_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["MLI_RESULT_LINE_EDIT_DEFAULT"])
        lines_label_and_spin_box = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["LINES_LABEL_TEXT"],
            self.box_config_data["NUMBER_OF_LINES_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["LINES_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["LINES_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["LINES_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["LINES_SPIN_BOX_STEP"],
            self.box_config_data["LINES_SPIN_BOX_SUFFIX"])
        min_length_label_and_spin_box = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["MIN_LENGTH_LABEL_TEXT"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_STEP"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_SUFFIX"])
        scale_label_and_spin_box = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["SCALE_LABEL_TEXT"],
            self.box_config_data["SCALE_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["SCALE_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["SCALE_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["SCALE_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["SCALE_SPIN_BOX_STEP"],
            self.box_config_data["SCALE_SPIN_BOX_SUFFIX"],
            "double")

        self.lines_label_and_spin_box_layout = lines_label_and_spin_box[0]
        self.min_length_label_and_spin_box_layout = min_length_label_and_spin_box[0]
        self.scale_label_and_spin_box_layout = scale_label_and_spin_box[0]

        self.lines_spin_box = lines_label_and_spin_box[2]
        self.min_length_spin_box = min_length_label_and_spin_box[2]
        self.scale_spin_box = scale_label_and_spin_box[2]

        ui_elements = [self.asvd_check_box_and_line_edit_layout,
                       horizontal_line_one,
                       self.mli_check_box_and_line_edit_layout,
                       self.lines_label_and_spin_box_layout,
                       self.min_length_label_and_spin_box_layout,
                       self.scale_label_and_spin_box_layout]

        self.create_action_box_layout(ui_elements,
                                      self.box_config_data["ACTION_BUTTON_TEXT"],
                                      self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"])

    def create_ui_rules(self):
        self.rules_engine.add_rule(lambda: self.mli_check_box.isChecked(),
                                   lambda: gui_creator.toggle(True, self.mli_line_edit))
        self.rules_engine.add_rule(lambda: self.asvd_check_box.isChecked(),
                                   lambda: gui_creator.toggle(True, self.asvd_line_edit))

        self.rules_engine.add_rule(lambda: not self.mli_check_box.isChecked(),
                                   lambda: gui_creator.toggle(False, self.mli_line_edit))
        self.rules_engine.add_rule(lambda: not self.asvd_check_box.isChecked(),
                                   lambda: gui_creator.toggle(False, self.asvd_line_edit))

        self.rules_engine.add_rule(lambda: not ActionBox.step == 2,
                                   lambda: gui_creator.toggle(False, self.action_button))
        self.rules_engine.add_rule(lambda: ActionBox.step == 3,
                                   lambda: gui_creator.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: ActionBox.step == 2,
                                    lambda: self.mli_check_box.isChecked() or self.asvd_check_box.isChecked()],
                                   lambda: gui_creator.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: ActionBox.step == 2,
                                    lambda: not self.mli_check_box.isChecked() and not self.asvd_check_box.isChecked()],
                                   lambda: gui_creator.toggle(False, self.action_button))

        self.rules_engine.add_rule(lambda: self.mli_check_box.isChecked(),
                                   lambda: gui_creator.toggle(True, [self.lines_spin_box,
                                                                                self.min_length_spin_box,
                                                                                self.scale_spin_box]))
        self.rules_engine.add_rule(lambda: not self.mli_check_box.isChecked(),
                                   lambda: gui_creator.toggle(False, [self.lines_spin_box,
                                                                                 self.min_length_spin_box,
                                                                                 self.scale_spin_box]))

        super().create_ui_rules()

    def on_results_ready(self, asvd, mli, chords, stdev_chord_lengths,
                         airspace_pixels, non_airspace_pixels, wrapped_assessments_layer):
        assessments_layer = wrapped_assessments_layer["assessments_layer"]

        gui_creator.update_line_edit(self.asvd_line_edit, asvd,
                                     self.box_config_data["ASVD_RESULT_LINE_EDIT_DEFAULT"], asvd)
        gui_creator.update_line_edit(self.mli_line_edit, mli,
                                     self.box_config_data["MLI_RESULT_LINE_EDIT_DEFAULT"], mli)

        if assessments_layer is not None:
            layers_editor.update_layers(self.napari_viewer,
                                        self.layers_config_data["ASSESSMENTS_LAYER"], assessments_layer,
                                        self.colormap_config_data, self.labels_config_data, True, False)

            self.napari_viewer.layers.selection.active = self.napari_viewer.layers[
                self.layers_config_data["POSTPROCESSING_LAYER"]]

        ActionBox.current_results = [
            os.path.basename(ActionBox.import_paths["image"]),
            ActionBox.current_use_computer_vision,
            os.path.basename(ActionBox.import_paths["weights"]),
            ActionBox.current_min_confidence,
            ActionBox.current_used_manual_threshold,
            ActionBox.current_threshold_value,
            ActionBox.current_remove_small_particles,
            ActionBox.current_remove_small_holes,
            self.asvd_check_box.isChecked(),
            self.mli_check_box.isChecked(),
            self.lines_spin_box.value(),
            self.min_length_spin_box.value(),
            self.scale_spin_box.value(),
            asvd,
            airspace_pixels,
            non_airspace_pixels,
            mli,
            stdev_chord_lengths,
            chords
        ]

        self.rules_engine.evaluate_rules()
        super().on_results_ready()


class ExportActionBox(ActionBox):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.mli_chords_line_edit = None
        self.mli_stdev_line_edit = None
        self.mli_line_edit = None

        self.mli_metrics = None

        self.asvd_non_airspace_pixels_line_edit = None
        self.asvd_airspace_pixels_line_edit = None
        self.asvd_line_edit = None

        self.asvd_metrics = None

        self.export_labelmap_check_box = None

        self.add_button = None
        self.remove_button = None
        self.clear_button = None

        self.accumulated_results = []

        self.accumulated_results: list[Result] = []
        self.current_result: Result | None = None

        self.exp_parent_folder = ""
        self.exp_project_name = ""
        self.exp_metrics_ext = "csv"
        self.exp_labelmap_ext = "tif"
        self.exp_rgb_color = False
        self.exp_zip_it = False

        self.box_id = 4

        self.create_ui_elements()
        self.create_ui_rules()

    def create_ui_elements(self):
        mli_layout, _, self.mli_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["MLI_METRIC"],
            self.box_config_data["MLI_METRIC_LINE_EDIT"]
        )
        mli_stdev_layout, _, self.mli_stdev_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["MLI_STDEV_METRIC"],
            self.box_config_data["MLI_STDEV_METRIC_LINE_EDIT"]
        )
        mli_chords_layout, _, self.mli_chords_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["MLI_CHORDS_METRIC"],
            self.box_config_data["MLI_CHORDS_METRIC_LINE_EDIT"]
        )

        self.mli_metrics = [
            self.mli_line_edit,
            self.mli_stdev_line_edit,
            self.mli_chords_line_edit
        ]
        hl1 = gui_creator.create_horizontal_line_widget()

        asvd_layout, _, self.asvd_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_METRIC"],
            self.box_config_data["ASVD_METRIC_LINE_EDIT"]
        )

        asvd_air_layout, _, self.asvd_airspace_pixels_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC"],
            self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC_LINE_EDIT"]
        )

        asvd_non_layout, _, self.asvd_non_airspace_pixels_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC"],
            self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC_LINE_EDIT"]
        )

        self.asvd_metrics = [
            self.asvd_line_edit,
            self.asvd_airspace_pixels_line_edit,
            self.asvd_non_airspace_pixels_line_edit
        ]
        hl2 = gui_creator.create_horizontal_line_widget()

        self.export_labelmap_check_box = gui_creator.create_check_box_widget(
            self.box_config_data["EXPORT_LABELMAP_CHECK_BOX_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["EXPORT_LABELMAP_CHECK_BOX_TOOLTIP_TEXT"],
            self.box_config_data["EXPORT_LABELMAP_CHECK_BOX_DEFAULT_VALUE"]
        )
        hl3 = gui_creator.create_horizontal_line_widget()

        add_layout, _, self.add_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["ADD_PROMPT"],
            self.box_config_data["ADD_BUTTON_TEXT"],
            self.box_config_data["ADD_BUTTON_TOOLTIP_TEXT"],
            self.add_results
        )
        rem_layout, _, self.remove_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["REMOVE_PROMPT"],
            self.box_config_data["REMOVE_BUTTON_TEXT"],
            self.box_config_data["REMOVE_BUTTON_TOOLTIP_TEXT"],
            self.remove_results
        )
        clr_layout, _, self.clear_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["CLEAR_PROMPT"],
            self.box_config_data["CLEAR_BUTTON_TEXT"],
            self.box_config_data["CLEAR_BUTTON_TOOLTIP_TEXT"],
            self.clear_results
        )

        rows = [
            asvd_layout,
            asvd_air_layout,
            asvd_non_layout,
            hl1,
            mli_layout,
            mli_stdev_layout,
            mli_chords_layout,
            hl2,
            self.export_labelmap_check_box,
            add_layout,
            hl3,
            rem_layout,
            clr_layout
        ]
        self.create_action_box_layout(
            rows,
            self.box_config_data["ACTION_BUTTON_TEXT"],
            self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"]
        )

    def create_ui_rules(self):
        self.rules_engine.add_rule([lambda: ActionBox.step == 3,
                                    lambda: ActionBox.current_results],
                                   [lambda: gui_creator.toggle(True, [self.add_button, self.export_labelmap_check_box]),
                                    lambda: self.set_results()])

        self.rules_engine.add_rule(lambda: self.mli_line_edit.text() == self.box_config_data["MLI_METRIC_LINE_EDIT"],
                                   lambda: gui_creator.toggle(False, self.mli_metrics))
        self.rules_engine.add_rule(lambda: self.asvd_line_edit.text() == self.box_config_data["ASVD_METRIC_LINE_EDIT"],
                                   lambda: gui_creator.toggle(False, self.asvd_metrics))
        self.rules_engine.add_rule(lambda: self.mli_line_edit.text() != self.box_config_data["MLI_METRIC_LINE_EDIT"],
                                   lambda: gui_creator.toggle(True, self.mli_metrics))
        self.rules_engine.add_rule(lambda: self.asvd_line_edit.text() != self.box_config_data["ASVD_METRIC_LINE_EDIT"],
                                   lambda: gui_creator.toggle(True, self.asvd_metrics))

        self.rules_engine.add_rule(lambda: not ActionBox.current_results,
                                lambda: gui_creator.toggle(False, [self.add_button, self.export_labelmap_check_box]))

        self.rules_engine.add_rule([lambda: ActionBox.current_results],
                                lambda: gui_creator.toggle(True, [self.add_button, self.export_labelmap_check_box]))

        self.rules_engine.add_rule([lambda: Result(*ActionBox.current_results) in self.accumulated_results],
                                lambda: gui_creator.toggle(False, [self.add_button, self.export_labelmap_check_box]))


        self.rules_engine.add_rule(lambda: len(self.accumulated_results) != 0,
                                   lambda: gui_creator.toggle(True, self.remove_button))
        self.rules_engine.add_rule(lambda: len(self.accumulated_results) == 0,
                                   lambda: gui_creator.toggle(False, self.remove_button))

        self.rules_engine.add_rule(lambda: not self.accumulated_results,
                                   lambda: gui_creator.toggle(False, self.clear_button))
        self.rules_engine.add_rule([lambda: self.accumulated_results],
                                   lambda: gui_creator.toggle(True, self.clear_button))

        self.rules_engine.add_rule(lambda: len(self.accumulated_results) != 0,
                                   lambda: gui_creator.toggle(True, self.action_button))
        self.rules_engine.add_rule(lambda: len(self.accumulated_results) == 0,
                                   lambda: gui_creator.toggle(False, self.action_button))

        super().create_ui_rules()

    def set_results(self):
        (
            _img, _cv, _w, _mc,
            _umt, _tv, _rsp, _rsh,
            asvd_on, mli_on,
            lines, min_len, scale,
            asvd, airspace, non_airspace,
            mli, stdev, chords
        ) = ActionBox.current_results

        self.asvd_line_edit.setText(str(asvd) if asvd_on else "None")
        self.asvd_airspace_pixels_line_edit.setText(str(airspace) if asvd_on else "None")
        self.asvd_non_airspace_pixels_line_edit.setText(str(non_airspace) if asvd_on else "None")

        self.mli_line_edit.setText(str(mli) if mli_on else "None")
        self.mli_stdev_line_edit.setText(str(stdev) if mli_on else "None")
        self.mli_chords_line_edit.setText(str(chords) if mli_on else "None")

        # _boxes.py, inside class ExportActionBox

    def add_results(self):
        raw = tuple(ActionBox.current_results)
        self.current_result = Result.from_raw(raw, labelmaps=None)
        if not self.current_result:
            return

        all_layer_names = [
            self.layers_config_data["INITIAL_LAYER"],
            self.layers_config_data["PROCESSING_LAYER"],
            self.layers_config_data["POSTPROCESSING_LAYER"],
            self.layers_config_data["ASSESSMENTS_LAYER"]
        ]

        labelmaps: Dict[str, np.ndarray] = {}
        if self.export_labelmap_check_box.isChecked():
            # ask for all four layers in order
            layers_data = layers_editor.get_layers_by_names(
                self.napari_viewer,
                all_layer_names
            )
            for layer_name, layer_data in zip(all_layer_names, layers_data):
                if layer_data is not None:
                    labelmaps[layer_name] = np.array(layer_data, copy=True)

        full_r = Result(
            **self.current_result.to_dict(),
            labelmaps=labelmaps or None
        )

        # store and update UI
        self.accumulated_results.append(full_r)
        self.rules_engine.evaluate_rules()
        self.update_export_counter()

    def remove_results(self):
        if gui_creator.create_confirmation_message_box(
                self, self.box_config_data["REMOVE_CONFIRMATION_MESSAGE"]
        ):
            self.accumulated_results.pop()
            self.rules_engine.evaluate_rules()
            self.update_export_counter()

    def clear_results(self):
        if gui_creator.create_confirmation_message_box(
                self, self.box_config_data["CLEAR_CONFIRMATION_MESSAGE"]
        ):
            self.accumulated_results.clear()
            self.rules_engine.evaluate_rules()
            self.update_export_counter()

    def update_export_counter(self):
        base = self.box_config_data["ACTION_BUTTON_TEXT"]
        n = len(self.accumulated_results)
        maxd = self.box_config_data["MAX_EXPORT_COUNT_DISPLAY_NUMBER"]

        if n == 0:
            txt = base
        elif n > maxd:
            txt = f"{base} ({maxd}+)"
        else:
            txt = f"{base} ({n})"

        self.action_button.setText(txt)

    def on_action_button_press(self):
        export_location = str(Path(ActionBox.import_paths["image"]).parent)

        if self.exp_parent_folder:
            export_location = self.exp_parent_folder

        if not is_real_writable_dir(export_location):
            export_location = self.box_config_data.get("DEFAULT_EXPORT_LOCATION", "")

        if not is_real_writable_dir(export_location):
            export_location = os.getcwd()

        has_labelmaps = any(r.labelmaps is not None for r in self.accumulated_results)
        params = gui_creator.get_export_params(self, export_location, has_labelmaps)

        if not params:
            return
        (
            self.exp_parent_folder,
            self.exp_project_name,
            self.exp_metrics_ext,
            self.exp_labelmap_ext,
            self.exp_rgb_color,
            self.exp_zip_it,
        ) = params

        super().on_action_button_press()

    def thread_worker(self):
        self.worker = ExportWorker()
        self.worker.set_parent_folder(self.exp_parent_folder)
        self.worker.set_project_name(self.exp_project_name)
        self.worker.set_metrics_format(self.exp_metrics_ext)
        self.worker.set_labelmap_format(self.exp_labelmap_ext)
        self.worker.set_exp_rgb(self.exp_rgb_color)
        self.worker.set_zip(self.exp_zip_it)
        self.worker.set_accumulated_results(self.accumulated_results)

        super().thread_worker()

    def on_thread_completed(self):
        super().on_thread_completed()
        self.update_export_counter()

    def on_results_ready(self, info: dict, error_msg: str):
        if error_msg:
            QMessageBox.critical(self, "Export Failed", error_msg)
            return

        msg = f"Metrics saved to:\n  {info['metrics']}"
        if info.get("labelmaps"):
            msg += f"\nLabelmaps folder:\n  {info['labelmaps']}"
        if info.get("archive"):
            msg += f"\nArchive file:\n  {info['archive']}"

        QMessageBox.information(self, "Export Complete", msg)
