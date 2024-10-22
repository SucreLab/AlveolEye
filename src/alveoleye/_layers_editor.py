from napari.utils.colormaps import DirectLabelColormap


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

        napari_viewer.add_labels(layer_data, colormap=colormap, opacity=1.0, name=layer_name)
        napari_viewer.layers[layer_name].editable = True
        return

    layer_data_rgb = layer_data[:, :, ::-1]
    napari_viewer.add_image(layer_data_rgb, name=layer_name)