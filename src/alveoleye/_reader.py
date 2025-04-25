import numpy as np
from typing import Optional, Callable, Any, Sequence
from typeguard import typechecked


@typechecked
def napari_get_reader(path: str | Sequence[str]) -> Optional[Callable[[str | Sequence[str]], list[tuple[np.ndarray, dict[str, Any], str]]]]:
    # This reader only opens .npy files
    if isinstance(path, (list, tuple)):
        first_path = path[0]
    else:
        first_path = path
    if not (str(first_path).endswith('.npy')):
        return None
    return reader_function


@typechecked
def reader_function(path: str | Sequence[str]) -> list[tuple[np.ndarray, dict[str, Any], str]]:
    paths = [path] if isinstance(path, str) else list(path)
    arrays = [np.load(_path) for _path in paths]
    data = np.squeeze(np.stack(arrays))
    add_kwargs: dict[str, Any] = {"name": "Initial"}
    return [(data, add_kwargs, "image")]