import os
import re

from IPython.external.qt_for_kernel import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

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

    spin_box_cls = NoScrollDoubleSpinBox if value_type == "double" else NoScrollSpinBox
    spin_box = spin_box_cls()
    spin_box.setMinimum(spin_box_min)
    spin_box.setMaximum(spin_box_max)
    spin_box.setValue(spin_box_default)
    spin_box.setSingleStep(spin_box_step)
    spin_box.setSuffix(spin_box_suffix)

    if value_type == "double":
        spin_box.setDecimals(decimals)

    label_and_spin_box_layout = create_sub_layout(QHBoxLayout(), [label, spin_box])

    spin_box.setToolTip(tooltip_text)
    spin_box.setCursor(Qt.PointingHandCursor)
    spin_box.setAlignment(QtCore.Qt.AlignCenter)

    spin_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return label_and_spin_box_layout, label, spin_box


def create_check_box_widget(check_box_text, on_check_box_checked, check_box_tooltip_text, default_checked=False):
    check_box = QCheckBox(check_box_text)
    check_box.setChecked(default_checked)
    check_box.stateChanged.connect(on_check_box_checked)
    check_box.setToolTip(check_box_tooltip_text)
    check_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    return check_box


def create_check_box_and_spin_box_layout(check_box_text, check_box_tooltip_text, spin_box_tooltip_text,
                                         on_check_box_checked, spin_box_min, spin_box_max, spin_box_default,
                                         spin_box_step, spin_box_suffix="", value_type="single", decimals=5):
    check_box = QCheckBox(check_box_text)
    check_box.stateChanged.connect(on_check_box_checked)

    spin_box_cls = NoScrollDoubleSpinBox if value_type == "double" else NoScrollSpinBox
    spin_box = spin_box_cls()
    spin_box.setMinimum(spin_box_min)
    spin_box.setMaximum(spin_box_max)
    spin_box.setValue(spin_box_default)
    spin_box.setSingleStep(spin_box_step)
    spin_box.setSuffix(spin_box_suffix)

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


def toggle(state, elements):
    if not isinstance(elements, list):
        elements = [elements]

    for item in elements:
        stack = [item]
        while stack:
            current = stack.pop()
            if isinstance(current, QLayout):
                for i in range(current.count()):
                    stack.append(current.itemAt(i).widget())
            else:
                current.setEnabled(state)


class ExportDialog(QDialog):
    def __init__(self, parent=None, default_parent_folder: str = None, has_labelmaps: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(400)

        self._default_parent = default_parent_folder or os.getcwd()
        self.has_labelmaps = has_labelmaps

        self.build_ui()

    def build_ui(self):
        self.parent_le = QLineEdit(self._default_parent)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_parent)

        h1 = QHBoxLayout()
        h1.addWidget(self.parent_le)
        h1.addWidget(browse)

        self.project_le = QLineEdit("results")

        self.metrics_combo = QComboBox(self)
        self.metrics_combo.addItems(["csv", "json"])

        form = QFormLayout()
        form.addRow("Parent folder:", h1)
        form.addRow("Project name:", self.project_le)
        form.addRow("Metrics format:", self.metrics_combo)

        if self.has_labelmaps:
            self.labelmap_combo = QComboBox(self)
            self.labelmap_combo.addItems(["tif", "png"])
            self.rgb_cb = QCheckBox("Export as RGB image")
            self.zip_cb = QCheckBox("Compress into ZIP archive")

            form.addRow("Labelmap format:", self.labelmap_combo)
            form.addRow("", self.rgb_cb)
            form.addRow("", self.zip_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addWidget(buttons)
        self.setLayout(v)

    def get_values(self):
        return (
            self.parent_le.text().strip(),
            self.project_le.text().strip(),
            self.metrics_combo.currentText(),
            self.labelmap_combo.currentText() if self.has_labelmaps else None,
            self.rgb_cb.isChecked() if self.has_labelmaps else False,
            self.zip_cb.isChecked() if self.has_labelmaps else False,
        )

    def _on_browse_parent(self):
        start = self.parent_le.text().strip() or self._default_parent
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Parent Folder",
            start,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if chosen:
            self.parent_le.setText(chosen)

    def _on_accept(self):
        p = self.parent_le.text().strip()
        if not p or not os.path.isdir(p) or not os.access(p, os.W_OK):
            QMessageBox.warning(self,
                                "Invalid Folder",
                                "Please pick an existing, writable folder.")
            return

        name = self.project_le.text().strip()
        if not name or re.search(r"[\\/]", name):
            QMessageBox.warning(self,
                                "Invalid Name",
                                "Enter a non‐empty name without slashes.")
            return

        self.accept()


def get_export_params(parent=None, default_parent_folder: str = None, has_labelmaps: bool = True):
    dlg = ExportDialog(parent, default_parent_folder=default_parent_folder, has_labelmaps=has_labelmaps)
    if dlg.exec_() == QDialog.Accepted:
        return dlg.get_values()
    return None
