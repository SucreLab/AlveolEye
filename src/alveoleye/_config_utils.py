from pathlib import Path
import json
from typing import Any, ClassVar
from typeguard import typechecked


@typechecked
class Config:
    _config_data: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def load(cls, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.json"
        with config_path.open('r', encoding='utf-8') as config_file:
            cls._config_data = json.load(config_file)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._config_data is None:
            cls.load()

    @classmethod
    def get_class_config(cls, klass: type) -> dict[str, Any]:
        section_name = klass.__name__
        return cls.get_section(section_name)

    @classmethod
    def get_section(cls, section_name: str) -> Any:
        cls._ensure_loaded()
        try:
            return cls._config_data[section_name]  # type: ignore
        except KeyError as e:
            raise KeyError(f"Section '{section_name}' not found in config file.") from e

    @classmethod
    def get_action_box(cls) -> dict[str, Any]:
        return cls.get_section("ActionBox")

    @classmethod
    def get_layers(cls) -> dict[str, Any]:
        return cls.get_section("Layers")

    @classmethod
    def get_labels(cls) -> dict[str, Any]:
        return cls.get_section("Labels")

    @classmethod
    def get_colormap(cls) -> dict[str, Any]:
        return cls.get_section("Colormap")

    @classmethod
    def get_label_indexed_colormap(cls) -> dict[Any, Any]:
        labels = cls.get_labels()
        colormap = cls.get_colormap()
        missing_keys = set(labels.keys()) - set(colormap.keys())
        if missing_keys:
            raise KeyError(f"Missing colormap entries for labels: {missing_keys}")
        return {labels[key]: colormap[key] for key in labels.keys()}

    @classmethod
    def get_snapshot_names(cls) -> dict[str, Any]:
        return cls.get_section("SnapshotNames")


__all__ = ["Config"]
