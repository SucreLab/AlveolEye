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


def update_layers(napari_viewer, layer_name, layer_data, is_labelmap):
    existing_layers = {layer.name: layer for layer in napari_viewer.layers}

    if layer_name in existing_layers:
        napari_viewer.layers.remove(existing_layers[layer_name])

    if is_labelmap:
        napari_viewer.add_labels(layer_data, opacity=1.0, name=layer_name)
        napari_viewer.layers[layer_name].editable = True
        # Todo: This should do something? Investigate.
        # napari_viewer.layers[layer_name].color_dict = {1: "red", 2: "green", 3: "green"}
        return

    layer_data_rgb = layer_data[:, :, ::-1]
    napari_viewer.add_image(layer_data_rgb, name=layer_name)
