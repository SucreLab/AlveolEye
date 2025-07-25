from pathlib import Path

from qtpy.QtCore import Qt, QTimer, QThread
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QGroupBox

import alveoleye._gui_creator
import alveoleye._rules as rules
import alveoleye._gui_creator as gui_creator
from alveoleye._config_utils import Config
from alveoleye._export_operations import make_save_image_callback
from alveoleye._workers import WorkerParent


class ActionBox(QGroupBox):
    current_results = []

    import_paths = {
        "image": None,
        "weights": None
    }

    # new shared state for export
    current_use_computer_vision: bool = False
    current_min_confidence: float = 0.0
    current_used_manual_threshold: bool = False
    current_threshold_value: float = 0.0
    current_remove_small_particles: int = False
    current_remove_small_holes: int = False

    all_action_boxes = []
    step = 0

    def __init__(self, napari_viewer):
        super().__init__()

        Config.load()

        self.action_box_config_data = Config.get_action_box()
        self.layers_config_data = Config.get_layers()
        self.labels_config_data = Config.get_labels()
        self.colormap_config_data = Config.get_label_indexed_colormap()
        self.box_config_data = Config.get_class_config(self.__class__)

        ActionBox.all_action_boxes.append(self)
        WorkerParent.layers_config_data = self.layers_config_data

        self.name = self.box_config_data["TITLE"]
        self.setTitle(self.name)

        self.state = 0

        self.layout = QVBoxLayout()
        self.rules_engine = rules.RulesEngine()

        self.worker = None
        self.thread = None
        self.action_button = None
        self.animation_timer = None

        self.box_id = None

        self.napari_viewer = napari_viewer

    class ActionButton(QPushButton):
        def __init__(self, button_text, tooltip_text, action, *__args):
            super().__init__(*__args)

            self.button_text = button_text
            self.tooltip_text = tooltip_text

            self.setText(button_text)
            self.setToolTip(tooltip_text)
            self.clicked.connect(action)

            self.setCursor(Qt.PointingHandCursor)

        def set_state(self, state):
            self.setText([self.button_text, "Cancel", "Canceling..."][state])
            self.setToolTip([self.tooltip_text, "Cancel operation", "Operation is canceling"][state])

    def broadcast_cancel_message(self):
        for box in ActionBox.all_action_boxes:
            if box is not self:
                box.cancel_action()

    def broadcast_step_change_message(self, box_id=None):
        for box in ActionBox.all_action_boxes:
            if box is not self:
                ActionBox.step = self.box_id if box_id is None else box_id
                box.rules_engine.evaluate_rules()

    def create_action_box_layout(self, elements, button_text, tooltip_text):
        self.action_button = self.ActionButton(button_text, tooltip_text, self.on_action_button_press)
        elements.append(self.action_button)
        gui_creator.create_sub_layout(self.layout, elements)
        self.setLayout(self.layout)

    def start_animation(self):
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.on_animation)
        self.animation_timer.start(self.action_box_config_data["ANIMATION_INTERVAL"])

    def stop_animation(self):
        self.setTitle(self.name)
        self.animation_timer.stop()

    def on_animation(self):
        current_dot_count = self.title().count(".") - self.name.count(".")
        new_dot_count = (current_dot_count + 1) % (self.action_box_config_data["ANIMATION_DOTS"] + 1)
        new_title = f"{self.name}{'.' * new_dot_count}"

        self.setTitle(new_title)

    def thread_worker(self):
        if self.action_box_config_data["SAVE_INTERMEDIATE_SNAPSHOTS"]:
            export_location = Path.home() / self.action_box_config_data["INTERMEDIATE_SNAPSHOT_SAVE_LOCATION"]
            self.worker.set_callback(make_save_image_callback(export_location))

        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.results_ready.connect(self.on_results_ready)
        self.worker.finished.connect(self.thread.quit)

        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.on_thread_completed)

        self.thread.start()

    def set_state(self, state):
        self.state = state
        self.action_button.set_state(state)
        self.rules_engine.evaluate_rules()

    def cancel_action(self):
        if self.state == 1:
            self.worker.cancel()
            self.set_state(2)

    def on_action_button_press(self):
        self.broadcast_cancel_message()

        if self.state:
            self.cancel_action()
            return

        self.start_animation()
        self.thread_worker()
        self.set_state(1)

    def on_results_ready(self, *__args):
        self.broadcast_step_change_message()

    def on_thread_completed(self):
        self.thread.deleteLater()
        self.stop_animation()
        self.set_state(0)

    def create_ui_rules(self):
        self.rules_engine.add_rule(lambda: self.state == 2, lambda: alveoleye._gui_creator.toggle(False, self.action_button))

        self.rules_engine.evaluate_rules()
