from typing import Mapping
import numpy as np
from napari.utils.colormaps import DirectLabelColormap
from napari.viewer import Viewer
from typeguard import typechecked


@typechecked
def get_layer_by_name(napari_viewer: Viewer, layer_name: str) -> np.ndarray | None:
    for layer in napari_viewer.layers:
        if layer.name == layer_name:
            return layer.data
    return None


@typechecked
def remove_layer(napari_viewer: Viewer, layer_name: str) -> None:
    layers_to_remove = list(napari_viewer.layers)
    for layer in layers_to_remove:
        if layer.name == layer_name:
            napari_viewer.layers.remove(layer)
            break


@typechecked
def remove_all_layers(napari_viewer: Viewer) -> None:
    layers_to_remove = list(napari_viewer.layers)
    for layer in layers_to_remove:
        napari_viewer.layers.remove(layer)


@typechecked
def _labels_dict_to_properties_array(labels_dict: Mapping[str, int]) -> list[str]:
    max_index = max(labels_dict.values())
    result_array: list[str] = ["undefined"] * (max_index + 1)
    for name, value in labels_dict.items():
        name_with_spaces = name.replace("_", " ").title()
        result_array[value] = name_with_spaces
    return result_array


@typechecked
def update_layers(
        napari_viewer: Viewer,
        layer_name: str,
        layer_data: np.ndarray,
        color_dict: Mapping[int | None, list[float]],
        labels_dict: Mapping[str, int],
        is_labelmap: bool
) -> None:
    existing_layers = {layer.name: layer for layer in napari_viewer.layers}
    if layer_name in existing_layers:
        napari_viewer.layers.remove(existing_layers[layer_name])
    if is_labelmap:
        color_dict = dict(color_dict)  # to allow item assignment if input is Mapping
        color_dict[None] = [0, 0, 0]
        colormap = DirectLabelColormap(color_dict=color_dict)
        properties = _labels_dict_to_properties_array(labels_dict)
        napari_viewer.add_labels(
            layer_data,
            colormap=colormap,
            properties=properties,
            opacity=1.0,
            name=layer_name
        )
        napari_viewer.layers[layer_name].editable = True
        return
    layer_data_rgb = layer_data[:, :, ::-1]
    napari_viewer.add_image(layer_data_rgb, name=layer_name)
