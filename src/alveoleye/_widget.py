import json
from pathlib import Path
from typing import Optional
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QWidget, QScrollArea
from napari.utils.theme import get_system_theme
from napari.viewer import Viewer  # Assuming this is used in napari plugins
from typeguard import typechecked
from alveoleye._boxes import (
    ProcessingActionBox,
    PostprocessingActionBox,
    AssessmentsActionBox,
    ExportActionBox,
)
import alveoleye._gui_creator as gui_creator

@typechecked
class WidgetMain(QWidget):
    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.processing_group_box: Optional[ProcessingActionBox] = None
        self.postprocessing_group_box: Optional[PostprocessingActionBox] = None
        self.assessments_group_box: Optional[AssessmentsActionBox] = None
        self.export_group_box: Optional[ExportActionBox] = None
        self.napari_viewer: Viewer = napari_viewer
        self.outer_widget: QWidget = QWidget()
        self.outer_layout: QVBoxLayout = QVBoxLayout(self.outer_widget)
        self.scroll_area: QScrollArea = QScrollArea()

        config_path: Path = Path(__file__).resolve().parent / "config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
            self.config_data: dict[str, object] = json.load(config_file)

        self.init_ui()

    def init_ui(self) -> None:
        self.create_action_boxes()
        self.setup_layout()
        self.apply_theme()

        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.outer_widget)

        main_layout: QVBoxLayout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)

        self.napari_viewer.events.theme.connect(self.apply_theme)

    def create_action_boxes(self) -> None:
        self.processing_group_box = ProcessingActionBox(self.config_data, self.napari_viewer)
        self.postprocessing_group_box = PostprocessingActionBox(self.config_data, self.napari_viewer)
        self.assessments_group_box = AssessmentsActionBox(self.config_data, self.napari_viewer)
        self.export_group_box = ExportActionBox(self.config_data, self.napari_viewer)

    def setup_layout(self) -> None:
        boxes: list[Optional[QWidget]] = [
            self.processing_group_box,
            self.postprocessing_group_box,
            self.assessments_group_box,
            self.export_group_box,
        ]
        gui_creator.create_sub_layout(self.outer_layout, boxes)

    def apply_theme(self) -> None:
        theme: str = (
            get_system_theme()
            if self.napari_viewer.theme == "system"
            else self.napari_viewer.theme
        )
        theme_file: str = "light_theme.css" if theme == "light" else "dark_theme.css"
        theme_path: Path = Path(__file__).resolve().parent / theme_file
        with theme_path.open("r", encoding="utf-8") as theme_file_handle:
            theme_css: str = theme_file_handle.read()
        self.setStyleSheet(theme_css)