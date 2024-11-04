import cv2
import os
from pathlib import Path

from qtpy.QtWidgets import QFileDialog

from alveoleye._action_box import ActionBox

from alveoleye._workers import ProcessingWorker, PostprocessingWorker, AssessmentsWorker, ExportWorker

import alveoleye._gui_creator as gui_creator
import alveoleye._layers_editor as layers_editor
import alveoleye._rules as rules


class ProcessingActionBox(ActionBox):
    model_output = None

    def __init__(self, config_data, napari_viewer):
        super().__init__(config_data, napari_viewer)

        self.image = None

        self.import_image_line_edit = None
        self.import_weights_line_edit = None
        self.confidence_threshold_spin_box = None
        self._image_open_path = None

        self.box_id = 1

        self.create_ui_elements()
        self.create_ui_rules()
        self.set_default_weights()

    def set_default_weights(self):
        default_file_name = self.box_config_data["DEFAULT_WEIGHTS_PATH"]
        default_weights_path = Path(__file__).resolve().parent.parent / default_file_name
        self.import_weights_line_edit.setText(Path(default_weights_path).name)
        ActionBox.import_paths["weights"] = default_weights_path

        self.rules_engine.evaluate_rules()

    def thread_worker(self):
        self.worker = ProcessingWorker()

        self.worker.set_napari_viewer(self.napari_viewer)
        self.worker.set_image_path(ActionBox.import_paths["image"])
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
        import_weights_button_and_line_edit = gui_creator.create_button_and_line_edit_layout(
            self.box_config_data["IMPORT_WEIGHTS_BUTTON_TEXT"],
            self.box_config_data["IMPORT_WEIGHTS_BUTTON_TOOLTIP_TEXT"],
            self.on_import_weights_press,
            self.box_config_data["EMPTY_PATH_LINE_EDIT_TEXT"]
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
        self.import_weights_line_edit = import_weights_button_and_line_edit[2]
        self.confidence_threshold_spin_box = confidence_threshold_label_and_spin_box[2]

        ui_elements = [import_image_button_and_line_edit_layout,
                       import_weights_button_and_line_edit_layout,
                       confidence_threshold_label_and_spin_box_layout]

        self.create_action_box_layout(ui_elements,
                                      self.box_config_data["ACTION_BUTTON_TEXT"],
                                      self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"])

    def create_ui_rules(self):
        self.rules_engine.add_rule([lambda: ActionBox.import_paths["image"] is None,
                                    lambda: ActionBox.import_paths["weights"] is None,
                                    lambda: not self.state == 2],
                                   lambda: rules.toggle(False, self.action_button))
        self.rules_engine.add_rule([lambda: ActionBox.import_paths["image"] is not None,
                                    lambda: ActionBox.import_paths["weights"] is not None],
                                   lambda: rules.toggle(True, self.action_button))

        self.rules_engine.add_rule(lambda: ActionBox.import_paths["image"] is None,
                                   lambda: rules.toggle(False, self.import_image_line_edit))
        self.rules_engine.add_rule(lambda: ActionBox.import_paths["image"] is not None,
                                   lambda: rules.toggle(True, self.import_image_line_edit))

        self.rules_engine.add_rule(lambda: ActionBox.import_paths["weights"] is None,
                                   lambda: rules.toggle(False, self.import_weights_line_edit))
        self.rules_engine.add_rule(lambda: ActionBox.import_paths["weights"] is not None,
                                   lambda: rules.toggle(True, self.import_weights_line_edit))

        super().create_ui_rules()

    def open_file_dialogue(self, title, accepted_extensions):
        if self._image_open_path is None:
            current_path = Path(__file__).resolve()
        else:
            current_path = self._image_open_path

        parent_directory = str(current_path.parent) + "/data"

        file_path = QFileDialog.getOpenFileName(self, title, parent_directory, accepted_extensions)[0]

        if file_path:
            file_name = Path(file_path).name
            self._image_open_path = Path(file_path)
        else:
            file_name = None

        return file_path, file_name

    def on_import_press(self, file_type, file_line_edit, dialogue_text, accepted_file_formats):
        file_path, file_name = self.open_file_dialogue(dialogue_text, accepted_file_formats)

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
                                    self.box_config_data["IMAGE_ACCEPTED_FILE_FORMATS"]):
            return
        try:
            self.image = cv2.imread(ActionBox.import_paths["image"])
        except Exception as e:
            print(f"Failed reading image {e}")
            self.broadcast_cancel_message()
            self.broadcast_step_change_message(0)
            return

        layers_editor.remove_all_layers(self.napari_viewer)
        layers_editor.update_layers(self.napari_viewer, self.layers_config_data["INITIAL_LAYER"], self.image,
                                    self.colormap_config_data, False)

        self.broadcast_cancel_message()
        self.broadcast_step_change_message(0)

    def on_import_weights_press(self):
        self.on_import_press("weights", self.import_weights_line_edit,
                             self.box_config_data["WEIGHTS_FILE_DIALOGUE_TEXT"],
                             self.box_config_data["WEIGHTS_ACCEPTED_FILE_FORMATS"])

    def on_results_ready(self, model_output, inference_labelmap):
        ProcessingActionBox.model_output = model_output

        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["ASSESSMENTS_LAYER_NAME"])
        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["POSTPROCESSING_LAYER"])
        layers_editor.update_layers(self.napari_viewer, self.layers_config_data["PROCESSING_LAYER"],
                                    inference_labelmap, self.colormap_config_data, True)

        super().on_results_ready()


class PostprocessingActionBox(ActionBox):
    def __init__(self, config_data, napari_viewer):
        super().__init__(config_data, napari_viewer)

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
        self.worker.set_model_output(ProcessingActionBox.model_output)
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
        self.rules_engine.add_rule(lambda: self.thresholding_check_box.isChecked(),
                                   lambda: rules.toggle(True, self.thresholding_spin_box))
        self.rules_engine.add_rule(lambda: not self.thresholding_check_box.isChecked(),
                                   lambda: rules.toggle(False, self.thresholding_spin_box))

        self.rules_engine.add_rule(lambda: ActionBox.step == 0,
                                   lambda: rules.toggle(False, self.action_button))

        self.rules_engine.add_rule([lambda: self.state == 0, lambda: ActionBox.step == 1],
                                   lambda: rules.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: self.state == 0, lambda: ActionBox.step == 2],
                                   lambda: rules.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: self.state == 0, lambda: ActionBox.step == 3],
                                   lambda: rules.toggle(True, self.action_button))

        super().create_ui_rules()

    def on_results_ready(self, labelmap):
        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["ASSESSMENTS_LAYER_NAME"])
        layers_editor.update_layers(self.napari_viewer, self.layers_config_data["POSTPROCESSING_LAYER"], labelmap,
                                    self.colormap_config_data, True)

        super().on_results_ready()


class AssessmentsActionBox(ActionBox):
    def __init__(self, config_data, napari_viewer):
        super().__init__(config_data, napari_viewer)

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
        self.worker.set_scale_spin_box_value(self.scale_spin_box.value())

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
                                   lambda: rules.toggle(True, self.mli_line_edit))
        self.rules_engine.add_rule(lambda: self.asvd_check_box.isChecked(),
                                   lambda: rules.toggle(True, self.asvd_line_edit))

        self.rules_engine.add_rule(lambda: not self.mli_check_box.isChecked(),
                                   lambda: rules.toggle(False, self.mli_line_edit))
        self.rules_engine.add_rule(lambda: not self.asvd_check_box.isChecked(),
                                   lambda: rules.toggle(False, self.asvd_line_edit))

        self.rules_engine.add_rule(lambda: not ActionBox.step == 2,
                                   lambda: rules.toggle(False, self.action_button))
        self.rules_engine.add_rule(lambda: ActionBox.step == 3,
                                   lambda: rules.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: ActionBox.step == 2,
                                    lambda: self.mli_check_box.isChecked() or self.asvd_check_box.isChecked()],
                                   lambda: rules.toggle(True, self.action_button))
        self.rules_engine.add_rule([lambda: ActionBox.step == 2,
                                    lambda: not self.mli_check_box.isChecked() and not self.asvd_check_box.isChecked()],
                                   lambda: rules.toggle(False, self.action_button))

        self.rules_engine.add_rule(lambda: self.mli_check_box.isChecked(),
                                   lambda: rules.toggle(True, [self.lines_spin_box,
                                                               self.min_length_spin_box,
                                                               self.scale_spin_box]))
        self.rules_engine.add_rule(lambda: not self.mli_check_box.isChecked(),
                                   lambda: rules.toggle(False, [self.lines_spin_box,
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
                                        self.layers_config_data["ASSESSMENTS_LAYER_NAME"], assessments_layer,
                                        self.colormap_config_data, True)

        ActionBox.current_results = [os.path.basename(ActionBox.import_paths["image"]),
                                     os.path.basename(ActionBox.import_paths["weights"]), asvd, mli,
                                     stdev_chord_lengths, chords, airspace_pixels, non_airspace_pixels,
                                     self.lines_spin_box.value(), self.min_length_spin_box.value(),
                                     self.scale_spin_box.value()]

        self.rules_engine.evaluate_rules()
        super().on_results_ready()


class ExportActionBox(ActionBox):
    def __init__(self, config_data, napari_viewer):
        super().__init__(config_data, napari_viewer)

        self.mli_chords_line_edit = None
        self.mli_stdev_line_edit = None
        self.mli_line_edit = None

        self.mli_metrics = None

        self.asvd_non_airspace_pixels_line_edit = None
        self.asvd_airspace_pixels_line_edit = None
        self.asvd_line_edit = None

        self.asvd_metrics = None

        self.add_button = None
        self.remove_button = None
        self.clear_button = None
        self.selected_filter = None
        self.file_path = None

        self.accumulated_results = []

        self.name_line_edit = None
        self.box_id = 4

        self.create_ui_elements()
        self.create_ui_rules()

    def on_action_button_press(self):
        self.file_path, self.selected_filter = gui_creator.save_data_with_file_dialog()
        super().on_action_button_press()

    def thread_worker(self):
        self.worker = ExportWorker()

        self.worker.set_file_path(self.file_path)
        self.worker.set_selected_filter(self.selected_filter)
        self.worker.set_accumulated_results(self.accumulated_results)

        super().thread_worker()

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

        self.mli_metrics = [self.mli_line_edit, self.mli_stdev_line_edit, self.mli_chords_line_edit]

        horizontal_line_one = gui_creator.create_horizontal_line_widget()

        asvd_layout, _, self.asvd_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_METRIC"],
            self.box_config_data["ASVD_METRIC_LINE_EDIT"]
        )

        (asvd_airspace_pixels_layout, _,
         self.asvd_airspace_pixels_line_edit) = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC"],
            self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC_LINE_EDIT"]
        )

        (asvd_non_airspace_pixels_layout, _,
         self.asvd_non_airspace_pixels_line_edit) = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC"],
            self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC_LINE_EDIT"]
        )

        self.asvd_metrics = [self.asvd_line_edit,
                             self.asvd_airspace_pixels_line_edit,
                             self.asvd_non_airspace_pixels_line_edit]

        horizontal_line_two = gui_creator.create_horizontal_line_widget()

        add_label_and_button_layout, label, self.add_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["ADD_PROMPT"],
            self.box_config_data["ADD_BUTTON_TEXT"],
            self.box_config_data["ADD_BUTTON_TOOLTIP_TEXT"],
            self.add_results)

        remove_label_and_button_layout, label, self.remove_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["REMOVE_PROMPT"],
            self.box_config_data["REMOVE_BUTTON_TEXT"],
            self.box_config_data["REMOVE_BUTTON_TOOLTIP_TEXT"],
            self.remove_results)

        clear_label_and_button_layout, label, self.clear_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["CLEAR_PROMPT"],
            self.box_config_data["CLEAR_BUTTON_TEXT"],
            self.box_config_data["CLEAR_BUTTON_TOOLTIP_TEXT"],
            self.clear_results)

        self.create_action_box_layout([asvd_layout, asvd_airspace_pixels_layout, asvd_non_airspace_pixels_layout,
                                       horizontal_line_one, mli_layout, mli_stdev_layout, mli_chords_layout,
                                       horizontal_line_two, add_label_and_button_layout, remove_label_and_button_layout,
                                       clear_label_and_button_layout], self.box_config_data["ACTION_BUTTON_TEXT"],
                                      self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"])

    def create_ui_rules(self):
        self.rules_engine.add_rule([lambda: ActionBox.step == 3,
                                    lambda: ActionBox.current_results],
                                   [lambda: rules.toggle(True, self.add_button),
                                    lambda: self.set_results()])

        self.rules_engine.add_rule(lambda: self.mli_line_edit.text() == self.box_config_data["MLI_METRIC_LINE_EDIT"],
                                   lambda: rules.toggle(False, self.mli_metrics))
        self.rules_engine.add_rule(lambda: self.asvd_line_edit.text() == self.box_config_data["ASVD_METRIC_LINE_EDIT"],
                                   lambda: rules.toggle(False, self.asvd_metrics))
        self.rules_engine.add_rule(lambda: self.mli_line_edit.text() != self.box_config_data["MLI_METRIC_LINE_EDIT"],
                                   lambda: rules.toggle(True, self.mli_metrics))
        self.rules_engine.add_rule(lambda: self.asvd_line_edit.text() != self.box_config_data["ASVD_METRIC_LINE_EDIT"],
                                   lambda: rules.toggle(True, self.asvd_metrics))

        self.rules_engine.add_rule(lambda: not ActionBox.current_results,
                                   lambda: rules.toggle(False, self.add_button))
        self.rules_engine.add_rule([lambda: ActionBox.current_results],
                                   lambda: rules.toggle(True, self.add_button))
        self.rules_engine.add_rule([lambda: tuple(ActionBox.current_results) in self.accumulated_results],
                                   lambda: rules.toggle(False, self.add_button))

        self.rules_engine.add_rule(lambda: len(self.accumulated_results) != 0,
                                   lambda: rules.toggle(True, self.remove_button))
        self.rules_engine.add_rule(lambda: len(self.accumulated_results) == 0,
                                   lambda: rules.toggle(False, self.remove_button))

        self.rules_engine.add_rule(lambda: not self.accumulated_results,
                                   lambda: rules.toggle(False, self.clear_button))
        self.rules_engine.add_rule([lambda: self.accumulated_results],
                                   lambda: rules.toggle(True, self.clear_button))

        self.rules_engine.add_rule(lambda: len(self.accumulated_results) != 0,
                                   lambda: rules.toggle(True, self.action_button))
        self.rules_engine.add_rule(lambda: len(self.accumulated_results) == 0,
                                   lambda: rules.toggle(False, self.action_button))

        super().create_ui_rules()

    def set_results(self):
        _, _, asvd, mli, stdev, chords, airspace_pixels, non_airspace_pixels, _, _, _ = ActionBox.current_results

        gui_creator.update_line_edit(self.mli_line_edit, mli,
                                     self.box_config_data["MLI_METRIC_LINE_EDIT"], mli)
        gui_creator.update_line_edit(self.mli_stdev_line_edit, stdev,
                                     self.box_config_data["MLI_STDEV_METRIC_LINE_EDIT"], mli)
        gui_creator.update_line_edit(self.mli_chords_line_edit, chords,
                                     self.box_config_data["MLI_CHORDS_METRIC_LINE_EDIT"], mli)
        gui_creator.update_line_edit(self.asvd_line_edit, asvd,
                                     self.box_config_data["ASVD_METRIC_LINE_EDIT"], asvd)
        gui_creator.update_line_edit(self.asvd_airspace_pixels_line_edit, airspace_pixels,
                                     self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC_LINE_EDIT"], asvd)
        gui_creator.update_line_edit(self.asvd_non_airspace_pixels_line_edit, non_airspace_pixels,
                                     self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC_LINE_EDIT"], asvd)

    def add_results(self):
        self.accumulated_results.append(tuple(ActionBox.current_results))
        self.rules_engine.evaluate_rules()
        self.update_export_counter()

    def remove_results(self):
        result = gui_creator.create_confirmation_message_box(self, self.box_config_data["REMOVE_CONFIRMATION_MESSAGE"])

        if result:
            self.accumulated_results.pop()
            self.rules_engine.evaluate_rules()
            self.update_export_counter()

    def clear_results(self):
        result = gui_creator.create_confirmation_message_box(self, self.box_config_data["CLEAR_CONFIRMATION_MESSAGE"])

        if result:
            self.accumulated_results.clear()
            self.rules_engine.evaluate_rules()
            self.update_export_counter()

    def update_export_counter(self):
        base_text = self.box_config_data["ACTION_BUTTON_TEXT"]
        max_export_count_display_number = self.box_config_data["MAX_EXPORT_COUNT_DISPLAY_NUMBER"]
        number_of_results = len(self.accumulated_results)

        if number_of_results == 0:
            self.action_button.setText(base_text)
        elif number_of_results > max_export_count_display_number:
            self.action_button.setText(f'{base_text} ({max_export_count_display_number}+)')
        else:
            self.action_button.setText(f'{base_text} ({number_of_results})')

    def on_thread_completed(self):
        super().on_thread_completed()
        self.update_export_counter()

    def on_results_ready(self, wrapped_data, extension):
        pass



