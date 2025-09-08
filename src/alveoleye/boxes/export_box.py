# boxes/export_box.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from .box_abstract import ActionBox
import alveoleye._gui_creator as gui_creator
import alveoleye._layers_editor as layers_editor
from alveoleye._export_operations import is_real_writable_dir
from alveoleye._models import Result
from alveoleye._rules import RulesEngine


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

    def build_worker(self):
        from alveoleye._workers import ExportWorker

        worker = ExportWorker()
        worker.set_parent_folder(self.exp_parent_folder)
        worker.set_project_name(self.exp_project_name)
        worker.set_metrics_format(self.exp_metrics_ext)
        worker.set_labelmap_format(self.exp_labelmap_ext)
        worker.set_exp_rgb(self.exp_rgb_color)
        worker.set_zip(self.exp_zip_it)
        worker.set_accumulated_results(self.accumulated_results)
        return worker

    def create_ui_elements(self) -> None:
        mli_layout, _, self.mli_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["MLI_METRIC"],
            self.box_config_data["MLI_METRIC_LINE_EDIT"],
        )
        mli_stdev_layout, _, self.mli_stdev_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["MLI_STDEV_METRIC"],
            self.box_config_data["MLI_STDEV_METRIC_LINE_EDIT"],
        )
        mli_chords_layout, _, self.mli_chords_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["MLI_CHORDS_METRIC"],
            self.box_config_data["MLI_CHORDS_METRIC_LINE_EDIT"],
        )

        self.mli_metrics = [self.mli_line_edit, self.mli_stdev_line_edit, self.mli_chords_line_edit]
        hl1 = gui_creator.create_horizontal_line_widget()

        asvd_layout, _, self.asvd_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_METRIC"],
            self.box_config_data["ASVD_METRIC_LINE_EDIT"],
        )
        asvd_air_layout, _, self.asvd_airspace_pixels_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC"],
            self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC_LINE_EDIT"],
        )
        asvd_non_layout, _, self.asvd_non_airspace_pixels_line_edit = gui_creator.create_label_and_line_edit_layout(
            self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC"],
            self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC_LINE_EDIT"],
        )

        self.asvd_metrics = [self.asvd_line_edit, self.asvd_airspace_pixels_line_edit, self.asvd_non_airspace_pixels_line_edit]
        hl2 = gui_creator.create_horizontal_line_widget()

        self.export_labelmap_check_box = gui_creator.create_check_box_widget(
            self.box_config_data["EXPORT_LABELMAP_CHECK_BOX_TEXT"],
            self.rules_engine.evaluate_rules,
            self.box_config_data["EXPORT_LABELMAP_CHECK_BOX_TOOLTIP_TEXT"],
            self.box_config_data["EXPORT_LABELMAP_CHECK_BOX_DEFAULT_VALUE"],
        )
        hl3 = gui_creator.create_horizontal_line_widget()

        add_layout, _, self.add_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["ADD_PROMPT"],
            self.box_config_data["ADD_BUTTON_TEXT"],
            self.box_config_data["ADD_BUTTON_TOOLTIP_TEXT"],
            self.add_results,
        )
        rem_layout, _, self.remove_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["REMOVE_PROMPT"],
            self.box_config_data["REMOVE_BUTTON_TEXT"],
            self.box_config_data["REMOVE_BUTTON_TOOLTIP_TEXT"],
            self.remove_results,
        )
        clr_layout, _, self.clear_button = gui_creator.create_label_and_button_layout(
            self.box_config_data["CLEAR_PROMPT"],
            self.box_config_data["CLEAR_BUTTON_TEXT"],
            self.box_config_data["CLEAR_BUTTON_TOOLTIP_TEXT"],
            self.clear_results,
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
            clr_layout,
        ]
        self.create_action_box_layout(
            rows,
            self.box_config_data["ACTION_BUTTON_TEXT"],
            self.box_config_data["ACTION_BUTTON_TOOLTIP_TEXT"],
        )

    def create_ui_rules(self) -> None:
        # When at step 3 and we have current results: show add controls and refresh metrics display
        self.rules_engine.run_actions_when_condition_is_true(
            RulesEngine.condition_all(lambda: ActionBox.step == 3, lambda: bool(ActionBox.current_results)),
            lambda: gui_creator.toggle(True, [self.add_button, self.export_labelmap_check_box]),
            self.set_results,
        )

        # Add button + labelmap checkbox visible whenever current results exist
        self.rules_engine.toggle_visibility_based_on_condition(
            lambda: bool(ActionBox.current_results),
            [self.add_button, self.export_labelmap_check_box],
        )

        # Prevent adding duplicates
        self.rules_engine.run_actions_when_condition_is_true(
            lambda: Result(*ActionBox.current_results) in self.accumulated_results,
            lambda: gui_creator.toggle(False, [self.add_button, self.export_labelmap_check_box]),
        )

        # Remove/Clear visibility mirrors whether anything is accumulated
        self.rules_engine.toggle_visibility_based_on_condition(
            lambda: len(self.accumulated_results) != 0,
            self.remove_button,
        )
        self.rules_engine.toggle_visibility_based_on_condition(
            lambda: bool(self.accumulated_results),
            self.clear_button,
        )

        # Export button mirrors presence of staged results
        self.rules_engine.toggle_visibility_based_on_condition(
            lambda: len(self.accumulated_results) != 0,
            self.action_button,
        )

        # Reset metric displays when no current results
        self.rules_engine.run_actions_when_condition_is_true(
            lambda: not ActionBox.current_results,
            lambda: self.asvd_line_edit.setText(self.box_config_data["ASVD_METRIC_LINE_EDIT"]),
            lambda: self.asvd_airspace_pixels_line_edit.setText(self.box_config_data["ASVD_AIRSPACE_PIXELS_METRIC_LINE_EDIT"]),
            lambda: self.asvd_non_airspace_pixels_line_edit.setText(self.box_config_data["ASVD_NON_AIRSPACE_PIXELS_METRIC_LINE_EDIT"]),
            lambda: self.mli_line_edit.setText(self.box_config_data["MLI_METRIC_LINE_EDIT"]),
            lambda: self.mli_stdev_line_edit.setText(self.box_config_data["MLI_STDEV_METRIC_LINE_EDIT"]),
            lambda: self.mli_chords_line_edit.setText(self.box_config_data["MLI_CHORDS_METRIC_LINE_EDIT"]),
        )

        super().create_ui_rules()

    def set_results(self) -> None:
        (
            _img, _cv, _w, _mc,
            _umt, _tv, _rsp, _rsh,
            asvd_on, mli_on,
            _lines, _min_len, _scale,
            asvd, airspace, non_airspace,
            mli, stdev, chords
        ) = ActionBox.current_results

        self.asvd_line_edit.setText(str(asvd) if asvd_on else "None")
        self.asvd_airspace_pixels_line_edit.setText(str(airspace) if asvd_on else "None")
        self.asvd_non_airspace_pixels_line_edit.setText(str(non_airspace) if asvd_on else "None")

        self.mli_line_edit.setText(str(mli) if mli_on else "None")
        self.mli_stdev_line_edit.setText(str(stdev) if mli_on else "None")
        self.mli_chords_line_edit.setText(str(chords) if mli_on else "None")

    def add_results(self) -> None:
        raw = tuple(ActionBox.current_results)
        self.current_result = Result.from_raw(raw, labelmaps=None)
        if not self.current_result:
            return

        all_layer_names = [
            self.layers_config_data["INITIAL_LAYER"],
            self.layers_config_data["PROCESSING_LAYER"],
            self.layers_config_data["POSTPROCESSING_LAYER"],
            self.layers_config_data["ASSESSMENTS_LAYER"],
        ]

        labelmaps: Dict[str, np.ndarray] = {}
        if self.export_labelmap_check_box.isChecked():
            layers_data = layers_editor.get_layers_by_names(self.napari_viewer, all_layer_names)
            for layer_name, layer_data in zip(all_layer_names, layers_data):
                if layer_data is not None:
                    labelmaps[layer_name] = np.array(layer_data, copy=True)

        full_r = Result(**self.current_result.to_dict(), labelmaps=labelmaps or None)
        self.accumulated_results.append(full_r)
        self.rules_engine.evaluate_rules()
        self.update_export_counter()

    def remove_results(self) -> None:
        if gui_creator.create_confirmation_message_box(
                self, self.box_config_data["REMOVE_CONFIRMATION_MESSAGE"]
        ):
            self.accumulated_results.pop()
            self.rules_engine.evaluate_rules()
            self.update_export_counter()

    def clear_results(self) -> None:
        if gui_creator.create_confirmation_message_box(
                self, self.box_config_data["CLEAR_CONFIRMATION_MESSAGE"]
        ):
            self.accumulated_results.clear()
            self.rules_engine.evaluate_rules()
            self.update_export_counter()

    def update_export_counter(self) -> None:
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

    def on_action_button_press(self) -> None:
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

        # Kick off worker/thread via base
        super().on_action_button_press()

    # Keep the original behavior: do NOT broadcast step change after export
    def on_results_ready(self, info: dict, error_msg: str) -> None:
        if error_msg:
            QMessageBox.critical(self, "Export Failed", error_msg)
            return

        msg = f"Metrics saved to:\n  {info['metrics']}"
        if info.get("labelmaps"):
            msg += f"\nLabelmaps folder:\n  {info['labelmaps']}"
        if info.get("archive"):
            msg += f"\nArchive file:\n  {info['archive']}"

        QMessageBox.information(self, "Export Complete", msg)
