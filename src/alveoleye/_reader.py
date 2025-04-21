import numpy as np
from typing import Optional, Union, List, Tuple, Dict
from typeguard import typechecked


@typechecked
def napari_get_reader(path: Union[str, List[str]]) -> Optional[callable]:
    # Todo: Turn this into .npy file reader for output results loading
    if isinstance(path, list):
        path = path[0]

    if not path.endswith((".jpeg", ".jpg", ".png", ".tif", ".tiff")):
        return None

    return reader_function


@typechecked
def reader_function(path: Union[str, List[str]]) -> List[Tuple[np.ndarray, Dict[str, str], str]]:
    # Todo: Make this function correctly for saved data
    paths: List[str] = [path] if isinstance(path, str) else path

    arrays: List[np.ndarray] = [np.load(_path) for _path in paths]
    data: np.ndarray = np.squeeze(np.stack(arrays))

    add_kwargs: Dict[str, str] = {"name": "Initial"}

    return [(data, add_kwargs, "image")]
