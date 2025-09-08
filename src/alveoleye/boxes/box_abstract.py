# alveoleye/_action_box.py
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtWidgets import QGroupBox, QPushButton, QVBoxLayout

import alveoleye._gui_creator as gui_creator
import alveoleye._rules as rules
from alveoleye._config_utils import Config
from alveoleye._export_operations import make_save_image_callback
from alveoleye._workers import WorkerParent


class BoxState(IntEnum):
    IDLE = 0
    RUNNING = 1
    CANCELING = 2

class _ActionBoxMeta(type(QGroupBox), ABCMeta):
    pass

class ActionBox(QGroupBox, metaclass=_ActionBoxMeta):
    current_results: ClassVar[list] = []
    import_paths: ClassVar[Dict[str, Optional[str]]] = {"image": None, "weights": None}

    current_use_computer_vision: ClassVar[bool] = False
    current_min_confidence: ClassVar[float] = 0.0
    current_used_manual_threshold: ClassVar[bool] = False
    current_threshold_value: ClassVar[float] = 0.0
    current_remove_small_particles: ClassVar[int] = 0
    current_remove_small_holes: ClassVar[int] = 0

    all_action_boxes: ClassVar[List["ActionBox"]] = []
    step: ClassVar[int] = 0

    def __init__(self, napari_viewer):
        super().__init__()

        Config.load()
        self.action_box_config_data: Dict[str, Any] = Config.get_action_box()
        self.layers_config_data: Dict[str, Any] = Config.get_layers()
        self.labels_config_data: Dict[str, Any] = Config.get_labels()
        self.colormap_config_data: Dict[str, Any] = Config.get_label_indexed_colormap()
        self.box_config_data: Dict[str, Any] = Config.get_class_config(self.__class__)

        ActionBox.all_action_boxes.append(self)
        WorkerParent.layers_config_data = self.layers_config_data

        self.name: str = self.box_config_data["TITLE"]
        self.setTitle(self.name)

        self.state: BoxState = BoxState.IDLE

        self.layout = QVBoxLayout()
        self.rules_engine = rules.RulesEngine()

        self.worker: Optional[WorkerParent] = None
        self.thread: Optional[QThread] = None
        self.action_button: Optional["ActionBox.ActionButton"] = None
        self.animation_timer: Optional[QTimer] = None

        self.box_id: Optional[int] = None
        self.napari_viewer = napari_viewer

    class ActionButton(QPushButton):
        def __init__(self, button_text: str, tooltip_text: str, action, *__args):
            super().__init__(*__args)
            self.button_text = button_text
            self.tooltip_text = tooltip_text
            self.setText(button_text)
            self.setToolTip(tooltip_text)
            self.clicked.connect(action)
            self.setCursor(Qt.PointingHandCursor)

        def set_state(self, state: BoxState) -> None:
            labels = {
                BoxState.IDLE: self.button_text,
                BoxState.RUNNING: "Cancel",
                BoxState.CANCELING: "Canceling...",
            }
            tips = {
                BoxState.IDLE: self.tooltip_text,
                BoxState.RUNNING: "Cancel operation",
                BoxState.CANCELING: "Operation is canceling",
            }
            self.setText(labels[state])
            self.setToolTip(tips[state])

    def broadcast_cancel_message(self, boxes=None) -> None:
        boxes = ActionBox.all_action_boxes if not boxes else boxes

        for box in boxes:
            box.cancel_action()

    def broadcast_update_step(self, box_id: Optional[int] = None) -> None:
        ActionBox.step = self.box_id if box_id is None else box_id

        self.evaluate_box_rules()
    
    def evaluate_box_rules(self, boxes=None):
        boxes = ActionBox.all_action_boxes if not boxes else boxes

        for box in boxes:
            box.rules_engine.evaluate_rules()

    def create_action_box_layout(self, elements, button_text: str, tooltip_text: str) -> None:
        self.action_button = self.ActionButton(button_text, tooltip_text, self.on_action_button_press)
        elements.append(self.action_button)
        gui_creator.create_sub_layout(self.layout, elements)
        self.setLayout(self.layout)

    def start_animation(self) -> None:
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.on_animation)
        self.animation_timer.start(self.action_box_config_data["ANIMATION_INTERVAL"])

    def stop_animation(self) -> None:
        self.setTitle(self.name)
        if self.animation_timer is not None:
            self.animation_timer.stop()
            self.animation_timer.deleteLater()
            self.animation_timer = None

    def on_animation(self) -> None:
        current_dot_count = self.title().count(".") - self.name.count(".")
        new_dot_count = (current_dot_count + 1) % (self.action_box_config_data["ANIMATION_DOTS"] + 1)
        self.setTitle(f"{self.name}{'.' * new_dot_count}")

    @abstractmethod
    def build_worker(self) -> WorkerParent:
        raise NotImplementedError

    def thread_worker(self) -> None:
        self.worker = self.build_worker()

        if self.action_box_config_data["SAVE_INTERMEDIATE_SNAPSHOTS"]:
            export_location = Path.home() / self.action_box_config_data["INTERMEDIATE_SNAPSHOT_SAVE_LOCATION"]
            self.worker.set_callback(make_save_image_callback(export_location))

        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)

        self.worker.results_ready.connect(self.on_results_ready)
        self.worker.finished.connect(self.thread.quit)

        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.on_thread_completed)

        self.thread.start()

    def set_state(self, state: BoxState) -> None:
        self.state = state
        if self.action_button is not None:
            self.action_button.set_state(state)
        self.rules_engine.evaluate_rules()

    def cancel_action(self) -> None:
        if self.state is BoxState.RUNNING and self.worker is not None:
            self.worker.cancel()
            self.set_state(BoxState.CANCELING)

    def on_action_button_press(self) -> None:
        self.broadcast_cancel_message()

        if self.state is not BoxState.IDLE:
            self.cancel_action()
            return

        self.start_animation()
        self.thread_worker()
        self.set_state(BoxState.RUNNING)

    @abstractmethod
    def on_results_ready(self, *args) -> None:
        self.broadcast_update_step()

    def on_thread_completed(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
        
        self.stop_animation()
        self.set_state(BoxState.IDLE)

    @abstractmethod
    def create_ui_rules(self) -> None:
        self.rules_engine.add_rule(
            lambda: self.state is BoxState.CANCELING,
            lambda: gui_creator.toggle(False, self.action_button),
        )
        self.rules_engine.evaluate_rules()

    @abstractmethod
    def create_ui_elements(self) -> None:
        ...
