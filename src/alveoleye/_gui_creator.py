from typing import Union, Callable, Tuple, List, Optional, Dict
from IPython.external.qt_for_kernel import QtCore
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QMessageBox, QLineEdit, QDoubleSpinBox, QSpinBox, QHBoxLayout,
    QSizePolicy, QCheckBox, QPushButton, QFileDialog, QLabel, QLayout, QWidget
)
from qtpy.QtCore import Qt
from typeguard import typechecked


@typechecked
def create_sub_layout(layout: QLayout, elements: List[Union[QWidget, QLayout]]) -> QLayout:
    for element in elements:
        if isinstance(element, QLayout):
            layout.addLayout(element)
        else:
            layout.addWidget(element)
    return layout


@typechecked
def create_label_and_spin_box_layout(
    label_text: str,
    tooltip_text: str,
    spin_box_min: Union[int, float],
    spin_box_max: Union[int, float],
    spin_box_default: Union[int, float],
    spin_box_step: Union[int, float],
    spin_box_suffix: str,
    value_type: str = "single",
    decimals: int = 5
) -> Tuple[QHBoxLayout, QLineEdit, Union[QDoubleSpinBox, QSpinBox]]:
    label = QLineEdit(label_text)
    label.setReadOnly(True)
    label.setObjectName("labelLineEdit")

    spin_box: Union[QDoubleSpinBox, QSpinBox]
    if value_type == "double":
        spin_box = QDoubleSpinBox()
        spin_box.setDecimals(decimals)
    else:
        spin_box = QSpinBox()

    spin_box.setMinimum(spin_box_min)
    spin_box.setMaximum(spin_box_max)
    spin_box.setValue(spin_box_default)
    spin_box.setSingleStep(spin_box_step)
    spin_box.setSuffix(spin_box_suffix)

    label_and_spin_box_layout = create_sub_layout(QHBoxLayout(), [label, spin_box])

    spin_box.setToolTip(tooltip_text)
    spin_box.setCursor(Qt.PointingHandCursor)
    spin_box.setAlignment(QtCore.Qt.AlignCenter)

    spin_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return label_and_spin_box_layout, label, spin_box


@typechecked
def create_check_box_and_spin_box_layout(
    check_box_text: str,
    check_box_tooltip_text: str,
    spin_box_tooltip_text: str,
    on_check_box_checked: Callable,
    spin_box_min: Union[int, float],
    spin_box_max: Union[int, float],
    spin_box_default: Union[int, float],
    spin_box_step: Union[int, float],
    spin_box_suffix: str = "",
    value_type: str = "single",
    decimals: int = 5
) -> Tuple[QHBoxLayout, QCheckBox, Union[QDoubleSpinBox, QSpinBox]]:
    check_box = QCheckBox(check_box_text)
    check_box.stateChanged.connect(on_check_box_checked)

    spin_box: Union[QDoubleSpinBox, QSpinBox]
    if value_type == "double":
        spin_box = QDoubleSpinBox()
        spin_box.setDecimals(decimals)
    else:
        spin_box = QSpinBox()

    spin_box.setMinimum(spin_box_min)
    spin_box.setMaximum(spin_box_max)
    spin_box.setValue(spin_box_default)
    spin_box.setSingleStep(spin_box_step)
    spin_box.setSuffix(spin_box_suffix)

    layout = create_sub_layout(QHBoxLayout(), [check_box, spin_box])

    check_box.setToolTip(check_box_tooltip_text)
    spin_box.setToolTip(spin_box_tooltip_text)

    spin_box.setCursor(Qt.PointingHandCursor)
    spin_box.setAlignment(QtCore.Qt.AlignCenter)

    check_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    spin_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return layout, check_box, spin_box


@typechecked
def create_check_box_and_line_edit_layout(
    check_box_text: str,
    tooltip_text: str,
    on_check_box_checked: Callable,
    line_edit_text: str
) -> Tuple[QHBoxLayout, QCheckBox, QLineEdit]:
    check_box = QCheckBox(check_box_text)
    check_box.stateChanged.connect(on_check_box_checked)
    check_box.setToolTip(tooltip_text)
    check_box.setChecked(True)

    line_edit = QLineEdit(line_edit_text)
    line_edit.setAlignment(QtCore.Qt.AlignCenter)
    line_edit.setReadOnly(True)

    layout = create_sub_layout(QHBoxLayout(), [check_box, line_edit])

    check_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    line_edit.setCursorPosition(0)

    return layout, check_box, line_edit


@typechecked
def create_button_and_line_edit_layout(
    button_text: str,
    tooltip_text: str,
    on_button_pressed: Callable,
    line_edit_text: str
) -> Tuple[QHBoxLayout, QPushButton, QLineEdit]:
    button = QPushButton(button_text)
    button.clicked.connect(on_button_pressed)
    button.setToolTip(tooltip_text)
    button.setCursor(Qt.PointingHandCursor)

    line_edit = QLineEdit(line_edit_text)
    line_edit.setReadOnly(True)
    line_edit.setAlignment(QtCore.Qt.AlignCenter)
    line_edit.setCursorPosition(0)

    layout = create_sub_layout(QHBoxLayout(), [button, line_edit])

    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return layout, button, line_edit


@typechecked
def create_label_and_button_layout(
    label_text: str,
    button_text: str,
    tooltip_text: str,
    on_button_pressed: Callable
) -> Tuple[QHBoxLayout, QLineEdit, QPushButton]:
    label = QLineEdit(label_text)
    label.setObjectName("labelLineEdit")
    label.setReadOnly(True)

    button = QPushButton(button_text)
    button.clicked.connect(on_button_pressed)
    button.setToolTip(tooltip_text)
    button.setCursor(Qt.PointingHandCursor)

    layout = create_sub_layout(QHBoxLayout(), [label, button])

    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return layout, label, button


@typechecked
def create_label_and_line_edit_layout(
    label_text: str,
    line_edit_text: str
) -> Tuple[QHBoxLayout, QLineEdit, QLineEdit]:
    label = QLineEdit(label_text)
    label.setObjectName("labelLineEdit")
    label.setReadOnly(True)

    line_edit = QLineEdit(line_edit_text)
    line_edit.setReadOnly(True)
    line_edit.setAlignment(QtCore.Qt.AlignCenter)
    line_edit.setCursorPosition(0)

    layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(line_edit)

    return layout, label, line_edit


@typechecked
def update_line_edit(line_edit: QLineEdit, value: str, default: str, condition: str) -> None:
    line_edit.setText(value if condition else default)


@typechecked
def create_horizontal_line_widget() -> QLabel:
    horizontal_line = QLabel()
    horizontal_line.setFixedHeight(1)
    horizontal_line.setObjectName("divider")
    return horizontal_line


@typechecked
def save_data_with_file_dialog() -> Tuple[str, str]:
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_dialog = QFileDialog()
    file_dialog.setOptions(options)
    file_dialog.setWindowTitle("Save Data")

    file_dialog.setDefaultSuffix("csv")
    file_dialog.setNameFilter("CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")

    file_path, selected_filter = file_dialog.getSaveFileName(
        None, 'Save File', "",
        "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
    )

    return file_path, selected_filter


@typechecked
def create_confirmation_message_box(parent: Optional[QWidget], message: str) -> bool:
    message_box = QMessageBox(parent)
    message_box.setIcon(QMessageBox.Warning)
    message_box.setText(
        f'<html><body style="font-weight: normal;">{message}</body></html>'
    )

    buttons = {
        "Yes": QMessageBox.AcceptRole,
        "No": QMessageBox.RejectRole
    }

    button_objects: Dict[str, QPushButton] = {}
    for text, role in buttons.items():
        button = message_box.addButton(text, role)
        button.setFixedSize(80, 25)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button_objects[text] = button

    message_box.setDefaultButton(button_objects["No"])
    message_box.exec_()
    clicked_button = message_box.clickedButton()

    return clicked_button == button_objects["Yes"]


@typechecked
def toggle(state: bool, elements: Union[QWidget, QLayout, List[Union[QWidget, QLayout]]]) -> None:
    if not isinstance(elements, list):
        elements = [elements]

    for item in elements:
        stack: List[Union[QWidget, QLayout]] = [item]
        while stack:
            current = stack.pop()
            if isinstance(current, QLayout):
                for i in range(current.count()):
                    widget = current.itemAt(i).widget()
                    if widget:
                        stack.append(widget)
            elif current is not None:
                current.setEnabled(state)
