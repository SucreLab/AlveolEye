from IPython.external.qt_for_kernel import QtCore
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QMessageBox
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QLineEdit, QDoubleSpinBox, QSpinBox, QHBoxLayout, QSizePolicy,
                            QCheckBox, QPushButton, QFileDialog, QLabel, QLayout)


def create_sub_layout(layout, elements):
    for element in elements:
        if isinstance(element, QLayout):
            layout.addLayout(element)
        else:
            layout.addWidget(element)
    return layout


def create_label_and_spin_box_layout(label_text, tooltip_text, spin_box_min, spin_box_max, spin_box_default,
                                     spin_box_step, spin_box_suffix, value_type="single", decimals=5):
    label = QLineEdit(label_text)
    label.setReadOnly(True)
    label.setObjectName("labelLineEdit")

    spin_box_parameters = {
        "minimum": spin_box_min,
        "maximum": spin_box_max,
        "value": spin_box_default,
        "singleStep": spin_box_step,
        "suffix": spin_box_suffix
    }
    spin_box = QDoubleSpinBox(**spin_box_parameters) if value_type == "double" else QSpinBox(**spin_box_parameters)
    if value_type == "double":
        spin_box.setDecimals(decimals)

    label_and_spin_box_layout = create_sub_layout(QHBoxLayout(), [label, spin_box])

    spin_box.setToolTip(tooltip_text)
    spin_box.setCursor(Qt.PointingHandCursor)
    spin_box.setAlignment(QtCore.Qt.AlignCenter)

    spin_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return label_and_spin_box_layout, label, spin_box


def create_check_box_and_spin_box_layout(check_box_text, check_box_tooltip_text, spin_box_tooltip_text,
                                         on_check_box_checked, spin_box_min, spin_box_max, spin_box_default,
                                         spin_box_step, spin_box_suffix="", value_type="single", decimals=5):
    check_box = QCheckBox(check_box_text)
    check_box.stateChanged.connect(on_check_box_checked)

    spin_box_parameters = {
        "minimum": spin_box_min,
        "maximum": spin_box_max,
        "value": spin_box_default,
        "singleStep": spin_box_step,
        "suffix": spin_box_suffix
    }
    spin_box = QDoubleSpinBox(**spin_box_parameters) if value_type == "double" else QSpinBox(**spin_box_parameters)
    if value_type == "double":
        spin_box.setDecimals(decimals)

    check_box_and_spin_box_layout = create_sub_layout(QHBoxLayout(), [check_box, spin_box])

    check_box.setToolTip(check_box_tooltip_text)
    spin_box.setToolTip(spin_box_tooltip_text)

    spin_box.setCursor(Qt.PointingHandCursor)
    spin_box.setAlignment(QtCore.Qt.AlignCenter)

    check_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    spin_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return check_box_and_spin_box_layout, check_box, spin_box


def create_check_box_and_line_edit_layout(check_box_text, tooltip_text, on_check_box_checked, line_edit_text):
    check_box = QCheckBox(check_box_text)
    check_box.stateChanged.connect(on_check_box_checked)
    check_box.setToolTip(tooltip_text)
    check_box.setChecked(True)

    line_edit = QLineEdit(line_edit_text)
    line_edit.setAlignment(QtCore.Qt.AlignCenter)
    line_edit.setReadOnly(True)

    check_box_and_line_edit_layout = create_sub_layout(QHBoxLayout(), [check_box, line_edit])

    check_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    line_edit.setCursorPosition(0)

    return check_box_and_line_edit_layout, check_box, line_edit


def create_button_and_line_edit_layout(button_text, tooltip_text, on_button_pressed, line_edit_text):
    button = QPushButton(button_text)
    button.clicked.connect(on_button_pressed)

    line_edit = QLineEdit(line_edit_text)
    line_edit.setReadOnly(True)

    button_and_line_edit_layout = create_sub_layout(QHBoxLayout(), [button, line_edit])

    button.setToolTip(tooltip_text)
    button.setCursor(Qt.PointingHandCursor)

    line_edit.setAlignment(QtCore.Qt.AlignCenter)
    line_edit.setCursorPosition(0)

    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return button_and_line_edit_layout, button, line_edit


def create_label_and_button_layout(label_text, button_text, tooltip_text, on_button_pressed):
    label = QLineEdit(label_text)
    label.setObjectName("labelLineEdit")
    label.setReadOnly(True)

    button = QPushButton(button_text)
    button.clicked.connect(on_button_pressed)
    button.setToolTip(tooltip_text)

    label_and_button_layout = create_sub_layout(QHBoxLayout(), [label, button])

    button.setCursor(Qt.PointingHandCursor)
    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return label_and_button_layout, label, button


def create_label_and_line_edit_layout(label_text, line_edit_text):
    label = QLineEdit(label_text)
    label.setObjectName("labelLineEdit")
    label.setReadOnly(True)

    line_edit = QLineEdit(line_edit_text)
    line_edit.setReadOnly(True)
    line_edit.setAlignment(QtCore.Qt.AlignCenter)
    line_edit.setCursorPosition(0)
    line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    label_and_line_edit_layout = QHBoxLayout()
    label_and_line_edit_layout.addWidget(label)
    label_and_line_edit_layout.addWidget(line_edit)

    return label_and_line_edit_layout, label, line_edit


def update_line_edit(line_edit, value, default, condition):
    if condition:
        line_edit.setText(value)
    else:
        line_edit.setText(default)


def create_horizontal_line_widget():
    horizontal_line = QLabel()
    horizontal_line.setFixedHeight(1)
    horizontal_line.setObjectName("divider")
    return horizontal_line


def save_data_with_file_dialog():
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_dialog = QFileDialog()
    file_dialog.setOptions(options)
    file_dialog.setWindowTitle("Save Data")

    file_dialog.setDefaultSuffix("csv")
    file_dialog.setNameFilter("CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")

    file_path, selected_filter = file_dialog.getSaveFileName(None, 'Save File', "",
                                                             "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)")
    return file_path, selected_filter


def create_confirmation_message_box(parent, message):
    message_box = QMessageBox(parent)
    message_box.setIcon(QMessageBox.Warning)
    message_box.setText(
        f'<html><body style="font-weight: normal;">{message}</body></html>')

    buttons = {
        "Yes": QMessageBox.AcceptRole,
        "No": QMessageBox.RejectRole
    }

    button_objects = {}

    for text, role in buttons.items():
        button = message_box.addButton(text, role)
        button.setFixedSize(80, 25)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button_objects[text] = button

    message_box.setDefaultButton(button_objects["No"])

    message_box.exec_()
    clicked_button = message_box.clickedButton()

    return clicked_button == button_objects["Yes"]