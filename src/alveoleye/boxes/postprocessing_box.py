# boxes/postprocessing_box.py
from __future__ import annotations

from .box_abstract import ActionBox, BoxState
import alveoleye._gui_creator as gui_creator
import alveoleye._layers_editor as layers_editor
from alveoleye._rules import RulesEngine


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

    def build_worker(self):
        from alveoleye._workers import PostprocessingWorker

        worker = PostprocessingWorker()
        worker.set_napari_viewer(self.napari_viewer)
        worker.set_layer_names(self.layers_config_data)
        worker.set_labels(self.labels_config_data)
        worker.set_thresholding_check_box_value(self.thresholding_check_box.isChecked())
        worker.set_manual_threshold_value(self.thresholding_spin_box.value())
        worker.set_alveoli_minimum_size(self.clean_alveoli_spin_box.value())
        worker.set_parenchyma_minimum_size(self.clean_parenchyma_spin_box.value())
        return worker

    def create_ui_elements(self) -> None:
        thresh_pair = gui_creator.create_check_box_and_spin_box_layout(
            self.box_config_data["THRESHOLDING_BOX_TEXT"],
            self.box_config_data["MANUAL_THRESHOLD_CHECKBOX_TOOLTIP_TEXT"],
            self.box_config_data["MANUAL_THRESHOLD_SPIN_BOX_TOOLTIP_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["THRESHOLDING_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["THRESHOLDING_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["THRESHOLDING_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["THRESHOLDING_SPIN_BOX_STEP"],
        )
        clean_alv = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["CLEAN_ALVEOLI_LABEL_TEXT"],
            self.box_config_data["REMOVE_SMALL_PARTICLES_TOOLTIP_TEXT"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_STEP"],
            self.box_config_data["CLEAN_ALVEOLI_SPIN_BOX_SUFFIX"],
        )
        clean_par = gui_creator.create_label_and_spin_box_layout(
            self.box_config_data["CLEAN_PARENCHYMA_LABEL_TEXT"],
            self.box_config_data["REMOVE_SMALL_HOLES_SPIN_BOX_TOOLTIP_TEXT"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_MIN_VALUE"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_MAX_VALUE"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_DEFAULT_VALUE"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_STEP"],
            self.box_config_data["CLEAN_PARENCHYMA_SPIN_BOX_SUFFIX"],
        )

        self.thresholding_check_box = thresh_pair[1]
        self.thresholding_spin_box = thresh_pair[2]
        self.clean_alveoli_spin_box = clean_alv[2]
        self.clean_parenchyma_spin_box = clean_par[2]

        ui_elements = [thresh_pair[0], clean_alv[0], clean_par[0]]

        self.create_action_box_layout(
            ui_elements,
            self.box_config_data["ACTION_BUTTON_TEXT"],
            self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"],
        )

    def create_ui_rules(self) -> None:
        from .box_abstract import ActionBox, BoxState
        from .postprocessing_box import PostprocessingActionBox

        self.rules_engine.run_actions_when_condition_is_true(
            RulesEngine.condition_all(
                lambda: PostprocessingActionBox.threshold_value is not None,
                lambda: ActionBox.step == 0
            ),
            lambda: self.thresholding_spin_box.setValue(PostprocessingActionBox.threshold_value),
        )

        self.rules_engine.toggle_visibility_based_on_checkbox_state(self.thresholding_check_box, self.thresholding_spin_box)
        self.rules_engine.toggle_visibility_when_all_conditions_are_true([lambda: self.state is BoxState.IDLE, lambda: ActionBox.step in (1, 2, 3)], self.action_button)

        super().create_ui_rules()

    def on_results_ready(self, labelmap) -> None:
        layers_editor.remove_layer(self.napari_viewer, self.layers_config_data["ASSESSMENTS_LAYER"])
        layers_editor.update_layers(
            self.napari_viewer,
            self.layers_config_data["POSTPROCESSING_LAYER"],
            labelmap,
            self.colormap_config_data,
            self.labels_config_data,
            True,
            True,
        )

        ActionBox.current_results = []
        super().on_results_ready()

        ActionBox.current_used_manual_threshold = self.thresholding_check_box.isChecked()
        if ActionBox.current_used_manual_threshold:
            ActionBox.current_threshold_value = self.thresholding_spin_box.value()
        else:
            ActionBox.current_threshold_value = PostprocessingActionBox.threshold_value

        ActionBox.current_remove_small_particles = self.clean_alveoli_spin_box.value()
        ActionBox.current_remove_small_holes = self.clean_parenchyma_spin_box.value()
