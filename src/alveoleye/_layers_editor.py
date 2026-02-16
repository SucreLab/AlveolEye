from napari.utils.colormaps import DirectLabelColormap

from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Union


def get_layers_by_names(
        viewer,
        layer_names: Union[str, Sequence[str]],
        callback: Optional[Callable[[Any, str], None]] = None,
        return_data: bool = True,
        on_missing: str = "none"
) -> Union[Any, List[Any], None]:
    """
    If layer_names is a str: return one array (or None).
    If layer_names is a sequence: return a list in the same order.
    """
    single_input = isinstance(layer_names, str)
    names = [layer_names] if single_input else list(layer_names)

    name2layer = {layer.name: layer for layer in viewer.layers}
    results = []
    for name in names:
        layer = name2layer.get(name)
        if layer is None:
            if on_missing == "error":
                raise KeyError(f"Layer {name!r} not found")
            results.append(None)
            continue
        if callback is not None:
            callback(layer.data, layer.name)
        results.append(layer.data if return_data else layer)

    # unwrap single
    if single_input:
        return results[0]
    return results


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


def _labels_dict_to_properties_array(labels_dict):
    max_index = max(labels_dict.values())
    result_array = ["undefined"] * (max_index + 1)

    for name, value in labels_dict.items():
        name_with_spaces = name.replace("_", " ").title()
        result_array[value] = name_with_spaces

    return result_array


def update_layers(
        napari_viewer,
        layer_name,
        layer_data,
        color_dict,
        labels_dict,
        is_labelmap,
        editable=True,
):
    existing_layers = {layer.name: layer for layer in napari_viewer.layers}
    if layer_name in existing_layers:
        napari_viewer.layers.remove(existing_layers[layer_name])
    if is_labelmap:
        color_dict[None] = [0, 0, 0]
        colormap = DirectLabelColormap(color_dict=color_dict)
        properties = _labels_dict_to_properties_array(labels_dict)
        napari_viewer.add_labels(
            layer_data,
            colormap=colormap,
            properties=properties,
            opacity=1.0,
            name=layer_name,
        )
        napari_viewer.layers[layer_name].editable = editable  # Set editable here
        return
    layer_data_rgb = layer_data[:, :, ::-1]
    napari_viewer.add_image(layer_data_rgb, name=layer_name)
