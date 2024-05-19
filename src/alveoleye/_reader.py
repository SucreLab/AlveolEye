import numpy as np


def napari_get_reader(path):
    # Todo: Turn this into .npy file reader for output results loading
    if isinstance(path, list):
        path = path[0]

    if not path.endswith((".jpeg", ".jpg", ".png", ".tif", ".tiff")):
        return None

    return reader_function


def reader_function(path):
    # Todo: Make this function correctly for saved data
    paths = [path] if isinstance(path, str) else path

    arrays = [np.load(_path) for _path in paths]
    data = np.squeeze(np.stack(arrays))

    add_kwargs = {"name": "Initial"}

    return [(data, add_kwargs, "image")]
