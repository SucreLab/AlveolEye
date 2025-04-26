from __future__ import annotations
from typing import Callable, Any
from collections.abc import Iterable
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, QThread
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QGroupBox

import numpy as np

import alveoleye._gui_creator
import alveoleye._rules as rules
import alveoleye._gui_creator as gui_creator
from alveoleye._workers import WorkerParent

from napari.viewer import Viewer

from typeguard import typechecked


@typechecked
class ActionBox(QGroupBox):
    current_results: list[Any] = []

    import_paths: dict[str, str | None] = {
        "image": None,
        "weights": None
    }

    all_action_boxes: list[ActionBox] = []
    step: int = 0

    def __init__(self, config_data: dict, napari_viewer: Viewer) -> None:
        super().__init__()

        self.action_box_config_data: dict = config_data["ActionBox"]
        self.layers_config_data: dict = config_data["Layers"]
        self.labels_config_data: dict = config_data["Labels"]
        self.colormap_config_data: dict = {
            self.labels_config_data[key]: config_data["Colormap"][key]
            for key in self.labels_config_data
        }

        self.box_config_data: dict = config_data[self.__class__.__name__]

        ActionBox.all_action_boxes.append(self)
        WorkerParent.layers_config_data = self.layers_config_data

        self.name: str = self.box_config_data["TITLE"]
        self.setTitle(self.name)

        self.state: int = 0

        self.layout: QVBoxLayout = QVBoxLayout()
        self.rules_engine: rules.RulesEngine = rules.RulesEngine()

        self.worker: WorkerParent | None = None
        self.thread: QThread | None = None
        self.action_button: ActionBox.ActionButton | None = None
        self.animation_timer: QTimer | None = None

        self.box_id: int | None = None
        self.napari_viewer: Viewer = napari_viewer

    @typechecked
    class ActionButton(QPushButton):
        def __init__(
            self,
            button_text: str,
            tooltip_text: str,
            action: Callable,
            *args: Any
        ) -> None:
            super().__init__(*args)

            self.button_text: str = button_text
            self.tooltip_text: str = tooltip_text

            self.setText(button_text)
            self.setToolTip(tooltip_text)
            self.clicked.connect(action)

            self.setCursor(Qt.PointingHandCursor)

        def set_state(self, state: int) -> None:
            self.setText([self.button_text, "Cancel", "Canceling..."][state])
            self.setToolTip([self.tooltip_text, "Cancel operation", "Operation is canceling"][state])

    def broadcast_cancel_message(self) -> None:
        for box in ActionBox.all_action_boxes:
            if box is not self:
                box.cancel_action()

    def broadcast_step_change_message(self, box_id: int | None = None) -> None:
        for box in ActionBox.all_action_boxes:
            if box is not self:
                ActionBox.step = self.box_id if box_id is None else box_id
                box.rules_engine.evaluate_rules()

    def create_action_box_layout(self, elements: list[Any], button_text: str, tooltip_text: str) -> None:
        self.action_button = self.ActionButton(button_text, tooltip_text, self.on_action_button_press)
        elements.append(self.action_button)
        gui_creator.create_sub_layout(self.layout, elements)
        self.setLayout(self.layout)

    def start_animation(self) -> None:
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.on_animation)
        self.animation_timer.start(self.action_box_config_data["ANIMATION_INTERVAL"])

    def stop_animation(self) -> None:
        self.setTitle(self.name)
        if self.animation_timer:
            self.animation_timer.stop()

    def on_animation(self) -> None:
        current_dot_count: int = self.title().count(".") - self.name.count(".")
        new_dot_count: int = (current_dot_count + 1) % (self.action_box_config_data["ANIMATION_DOTS"] + 1)
        new_title: str = f"{self.name}{'.' * new_dot_count}"
        self.setTitle(new_title)

    def thread_worker(self) -> None:
        if self.worker is None:
            return

        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.results_ready.connect(self.on_results_ready)
        self.worker.finished.connect(self.thread.quit)

        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.on_thread_completed)

        self.thread.start()

    def set_state(self, state: int) -> None:
        self.state = state

        if self.action_button:
            self.action_button.set_state(state)

        self.rules_engine.evaluate_rules()

    def cancel_action(self) -> None:
        if self.state == 1 and self.worker:
            self.worker.cancel()
            self.set_state(2)

    def on_action_button_press(self) -> None:
        self.broadcast_cancel_message()

        if self.state:
            self.cancel_action()
            return

        self.start_animation()
        self.thread_worker()
        self.set_state(1)

    def on_results_ready(self, *args: Any) -> None:
        self.broadcast_step_change_message()

    def on_thread_completed(self) -> None:
        if self.thread:
            self.thread.deleteLater()
        self.stop_animation()
        self.set_state(0)

    def create_ui_rules(self) -> None:
        self.rules_engine.add_rule(
            lambda: self.state == 2,
            lambda: alveoleye._gui_creator.toggle(False, self.action_button)
        )
        self.rules_engine.evaluate_rules()
