import json
import pathlib

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QWidget, QScrollArea
from napari.utils.theme import get_system_theme
from typing import Union

from alveoleye._boxes import (ProcessingActionBox, PostprocessingActionBox,
                                               AssessmentsActionBox, ExportActionBox)
import alveoleye._gui_creator as gui_creator


class WidgetMain(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.assessments_group_box: Union[AssessmentsActionBox, None] = None
        self.postprocessing_group_box: Union[PostprocessingActionBox, None] = None
        self.processing_group_box: Union[ProcessingActionBox, None] = None
        self.export_group_box: Union[ExportActionBox, None] = None

        self.napari_viewer = napari_viewer

        self.outer_widget = QWidget()
        self.outer_layout = QVBoxLayout(self.outer_widget)
        self.scroll_area = QScrollArea()

        with open(pathlib.Path(__file__).resolve().parent / "config.json", 'r') as config_file:
            self.config_data = json.load(config_file)

        self.init_ui()

    def init_ui(self):
        self.create_action_boxes()
        self.setup_layout()
        self.apply_theme()

        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.outer_widget)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

        self.napari_viewer.events.theme.connect(self.apply_theme)

    def create_action_boxes(self):
        self.processing_group_box = ProcessingActionBox(self.config_data, self.napari_viewer)
        self.postprocessing_group_box = PostprocessingActionBox(self.config_data, self.napari_viewer)
        self.assessments_group_box = AssessmentsActionBox(self.config_data, self.napari_viewer)
        self.export_group_box = ExportActionBox(self.config_data, self.napari_viewer)

    def setup_layout(self):
        boxes = [self.processing_group_box, self.postprocessing_group_box,
                 self.assessments_group_box, self.export_group_box]
        gui_creator.create_sub_layout(self.outer_layout, boxes)

    def apply_theme(self):
        theme = get_system_theme() if self.napari_viewer.theme == "system" else self.napari_viewer.theme
        theme_file = "light_theme.css" if theme == "light" else "dark_theme.css"
        theme_path = pathlib.Path(__file__).resolve().parent / theme_file

        self.setStyleSheet(open(theme_path).read())
