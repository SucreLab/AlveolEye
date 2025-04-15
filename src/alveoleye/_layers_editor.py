from napari.utils.colormaps import DirectLabelColormap
from napari import Viewer
import numpy as np
from qtpy.QtWidgets import QLabel, QSizePolicy
from qtpy.QtCore import Qt
import warnings


def get_layer_by_name(napari_viewer, layer_name):
    for layer in napari_viewer.layers:
        if layer.name == layer_name:
            return layer.data

    return None


def remove_layer(napari_viewer, layer_name):
    layers_to_remove = list(napari_viewer.layers)
    for layer in layers_to_remove:
        if layer.name == layer_name:
            napari_viewer.layers.remove(layer)
            break


def remove_all_layers(napari_viewer):
    layers_to_remove = list(napari_viewer.layers)
    for layer in layers_to_remove:
        napari_viewer.layers.remove(layer)


def update_layers(napari_viewer, layer_name, layer_data, color_dict, is_labelmap):
    existing_layers = {layer.name: layer for layer in napari_viewer.layers}

    if layer_name in existing_layers:
        napari_viewer.layers.remove(existing_layers[layer_name])

    if is_labelmap:
        color_dict[None] = [0, 0, 0]
        colormap = DirectLabelColormap(color_dict=color_dict)

        layer = napari_viewer.add_labels(layer_data, colormap=colormap, opacity=1.0, name=layer_name)
        layer.mouse_move_callbacks.append(on_mouse_move_factory(napari_viewer))
        napari_viewer.layers[layer_name].editable = True
        return

    layer_data_rgb = layer_data[:, :, ::-1]
    layer = napari_viewer.add_image(layer_data_rgb, name=layer_name)
    layer.mouse_move_callbacks.append(on_mouse_move_factory(napari_viewer))


def on_mouse_move_factory(napari_viewer):
    def on_mouse_move(layer, event):
        tooltip = QLabel(napari_viewer.window.qt_viewer.parent())
        tooltip.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        tooltip.setAttribute(Qt.WA_ShowWithoutActivating)
        tooltip.setFixedSize(20, 20)
        tooltip.setAlignment(Qt.AlignCenter)
        tooltip.sizePolicy().setHorizontalPolicy(QSizePolicy.Fixed)
        tooltip.setStyleSheet("color: black")
        tooltip.show()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pos = napari_viewer.window.qt_viewer.cursor().pos()
            tooltip.move(pos.x()+20, pos.y()+20)
            val = layer.get_value(event.position)
            tooltip.setText(str(val) if val is not None else "-")

    return on_mouse_move
