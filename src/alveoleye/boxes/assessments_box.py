# boxes/assesment_box.py
from __future__ import annotations

import os

from .box_abstract import ActionBox, BoxState
import alveoleye._gui_creator as gui_creator
import alveoleye._layers_editor as layers_editor
from alveoleye._rules import RulesEngine


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

    def build_worker(self):
        from alveoleye._workers import AssessmentsWorker

        worker = AssessmentsWorker()
        worker.set_napari_viewer(self.napari_viewer)
        worker.set_layer_names(self.layers_config_data)
        worker.set_labels(self.labels_config_data)
        worker.set_asvd_check_box_state(self.asvd_check_box.isChecked())
        worker.set_mli_check_box_state(self.mli_check_box.isChecked())
        worker.set_lines_spin_box_value(self.lines_spin_box.value())
        worker.set_min_length_spin_box_value(self.min_length_spin_box.value())
        worker.set_scale_spin_box_value(round(self.scale_spin_box.value(), 5))
        return worker

    def create_ui_elements(self) -> None:
        (self.asvd_check_box_and_line_edit_layout,
         self.asvd_check_box,
         self.asvd_line_edit) = gui_creator.create_check_box_and_line_edit_layout(
            self.box_config_data["ASVD_CHECK_BOX_TITLE"],
            self.box_config_data["ASVD_CHECKBOX_TOOLTIP_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["ASVD_RESULT_LINE_EDIT_DEFAULT"])

        horizontal_line = gui_creator.create_horizontal_line_widget()

        (self.mli_check_box_and_line_edit_layout,
         self.mli_check_box,
         self.mli_line_edit) = gui_creator.create_check_box_and_line_edit_layout(
            self.box_config_data["MLI_CHECK_BOX_TITLE"],
            self.box_config_data["MLI_CHECKBOX_TOOLTIP_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["MLI_RESULT_LINE_EDIT_DEFAULT"])

        lines_lbl = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["LINES_LABEL_TEXT"],
            self.box_config_data["NUMBER_OF_LINES_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["LINES_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["LINES_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["LINES_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["LINES_SPIN_BOX_STEP"],
            self.box_config_data["LINES_SPIN_BOX_SUFFIX"])

        min_len_lbl = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["MIN_LENGTH_LABEL_TEXT"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_STEP"],
            self.box_config_data["MIN_LENGTH_SPIN_BOX_SUFFIX"])

        scale_lbl = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["SCALE_LABEL_TEXT"],
            self.box_config_data["SCALE_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["SCALE_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["SCALE_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["SCALE_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["SCALE_SPIN_BOX_STEP"],
            self.box_config_data["SCALE_SPIN_BOX_SUFFIX"],
            "double")

        self.lines_label_and_spin_box_layout = lines_lbl[0]
        self.min_length_label_and_spin_box_layout = min_len_lbl[0]
        self.scale_label_and_spin_box_layout = scale_lbl[0]

        self.lines_spin_box = lines_lbl[2]
        self.min_length_spin_box = min_len_lbl[2]
        self.scale_spin_box = scale_lbl[2]

        ui_elements = [
            self.asvd_check_box_and_line_edit_layout,
            horizontal_line,
            self.mli_check_box_and_line_edit_layout,
            self.lines_label_and_spin_box_layout,
            self.min_length_label_and_spin_box_layout,
            self.scale_label_and_spin_box_layout,
        ]

        self.create_action_box_layout(
            ui_elements,
            self.box_config_data["ACTION_BUTTON_TEXT"],
            self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"],
        )

    def create_ui_rules(self) -> None:
        from .box_abstract import ActionBox

        self.rules_engine.toggle_visibility_based_on_checkbox_state(self.mli_check_box,  self.mli_line_edit)
        self.rules_engine.toggle_visibility_based_on_checkbox_state(self.asvd_check_box, self.asvd_line_edit)

        self.rules_engine.toggle_visibility_based_on_condition(
            RulesEngine.condition_either(
                lambda: ActionBox.step == 3,
                RulesEngine.condition_all(
                    lambda: ActionBox.step == 2,
                    lambda: self.mli_check_box.isChecked() or self.asvd_check_box.isChecked(),
                ),
            ),
            self.action_button,
        )

        # MLI parameter widgets enabled/disabled with MLI checkbox
        self.rules_engine.enable_or_disable_based_on_checkbox_state(self.mli_check_box, [self.lines_spin_box, self.min_length_spin_box, self.scale_spin_box])

        # Reset result fields if there are no current results
        self.rules_engine.run_actions_when_condition_is_true(
            lambda: not ActionBox.current_results,
            lambda: self.asvd_line_edit.setText(self.box_config_data["ASVD_RESULT_LINE_EDIT_DEFAULT"]),
            lambda: self.mli_line_edit.setText(self.box_config_data["MLI_RESULT_LINE_EDIT_DEFAULT"]),
        )

        super().create_ui_rules()

    def on_results_ready(self, asvd, mli, chords, stdev_chord_lengths,
                         airspace_pixels, non_airspace_pixels, wrapped_assessments_layer) -> None:
        assessments_layer = wrapped_assessments_layer["assessments_layer"]

        gui_creator.update_line_edit(self.asvd_line_edit, asvd,
                                     self.box_config_data["ASVD_RESULT_LINE_EDIT_DEFAULT"], asvd)
        gui_creator.update_line_edit(self.mli_line_edit, mli,
                                     self.box_config_data["MLI_RESULT_LINE_EDIT_DEFAULT"], mli)

        if assessments_layer is not None:
            layers_editor.update_layers(
                self.napari_viewer,
                self.layers_config_data["ASSESSMENTS_LAYER"],
                assessments_layer,
                self.colormap_config_data,
                self.labels_config_data,
                True,
                False,
            )
            # Keep existing UX focus behavior
            self.napari_viewer.layers.selection.active = self.napari_viewer.layers[
                self.layers_config_data["POSTPROCESSING_LAYER"]
            ]

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
            chords,
        ]

        self.rules_engine.evaluate_rules()
        super().on_results_ready()
