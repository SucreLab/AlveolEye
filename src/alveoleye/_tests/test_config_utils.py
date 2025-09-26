import json
from pathlib import Path
import tempfile
import pytest

from alveoleye._config_utils import Config


def make_temp_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_and_get_section(tmp_path: Path, monkeypatch):
    data = {
        "ActionBox": {"TITLE": "Test Action Box"},
        "Layers": {"image": 0},
        "Labels": {"A": 1, "B": 2},
        "Colormap": {"A": [255, 0, 0, 255], "B": [0, 255, 0, 255]},
        "SnapshotNames": {"processing": "proc"},
    }
    cfg_path = make_temp_config(tmp_path, data)

    # Ensure we don't use previously cached config
    Config._config_data = None  # type: ignore
    Config.load(cfg_path)

    assert Config.get_section("ActionBox")["TITLE"] == "Test Action Box"
    assert Config.get_layers()["image"] == 0
    assert Config.get_action_box()["TITLE"] == "Test Action Box"
    assert Config.get_snapshot_names()["processing"] == "proc"


def test_get_class_config_uses_class_name(tmp_path: Path):
    data = {"DummyClass": {"X": 42}}
    cfg_path = make_temp_config(tmp_path, data)

    class DummyClass:
        pass

    Config._config_data = None  # type: ignore
    Config.load(cfg_path)
    assert Config.get_class_config(DummyClass)["X"] == 42


def test_missing_section_raises_keyerror(tmp_path: Path):
    data = {"Existing": {"Y": 1}}
    cfg_path = make_temp_config(tmp_path, data)

    Config._config_data = None  # type: ignore
    Config.load(cfg_path)

    with pytest.raises(KeyError):
        Config.get_section("NotThere")


def test_get_label_indexed_colormap_success(tmp_path: Path):
    data = {
        "Labels": {"A": 1, "B": 2},
        "Colormap": {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]},
    }
    cfg_path = make_temp_config(tmp_path, data)

    Config._config_data = None  # type: ignore
    Config.load(cfg_path)

    label_cmap = Config.get_label_indexed_colormap()
    assert label_cmap == {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}


def test_get_label_indexed_colormap_missing_key_raises(tmp_path: Path):
    data = {
        "Labels": {"A": 1, "B": 2},
        # Missing 'B' in Colormap
        "Colormap": {"A": [1, 2, 3, 4]},
    }
    cfg_path = make_temp_config(tmp_path, data)

    Config._config_data = None  # type: ignore
    Config.load(cfg_path)

    with pytest.raises(KeyError):
        Config.get_label_indexed_colormap()
