import dataclasses
from collections import namedtuple
from dataclasses import field, fields
from typing import Dict, List

Range = namedtuple("Range", ["min", "max"])


@dataclasses.dataclass
class Config:
    """
    Base Config class for storing latents, their types and ranges. Latent's type and size are should be defined as metadata.
    rv_type metadata field should be continuous, discrete or categorical; it is used later to adjust sampling strategies.
    """

    def __getitem__(self, item):
        return getattr(self, item)

    def __post_init__(self):
        for config_field in fields(self):
            assert config_field.metadata.get("rv_type") in [
                "continuous",
                "discrete",
                "categorical",
            ], f"rv_type for {config_field.name} is not in [continuous, discrete, categorical]"

    @property
    def get_total_latent_dim(self) -> int:
        count = 0
        for config_field in fields(self):
            if config_field.metadata.get("latent_size"):
                count += 1 * config_field.metadata.get("latent_size")
        return count

    def get_latents_metadata(self) -> Dict[str, str]:
        return {
            config_field.name: (
                config_field.metadata.get("rv_type"),
                config_field.metadata.get("latent_size"),
            )
            for config_field in fields(self)
            if config_field.metadata.get("rv_type")
            and config_field.metadata.get("latent_size")
        }


@dataclasses.dataclass
class SpriteWorldConfig(Config):
    """
    Config class for SpriteWorld dataset.
    """

    x: Range = field(
        default=Range(0, 1), metadata={"rv_type": "continuous", "latent_size": 10}
    )
    y: Range = field(
        default=Range(0, 1), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    shape: List[str] = field(
        default_factory=lambda: ["triangle", "square", "circle"],
        metadata={"rv_type": "categorical", "latent_size": 1},
    )
    scale: Range = field(
        default=Range(0, 1), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    angle: Range = field(
        default=Range(0, 360), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    colour_ch_0: Range = field(
        default=Range(0, 1), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    colour_ch_1: Range = field(
        default=Range(0, 1), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    colour_ch_2: Range = field(
        default=Range(0, 1), metadata={"rv_type": "continuous", "latent_size": 1}
    )

    def shape_to_idx(self, shape):
        if shape == "triangle":
            return 0
        elif shape == "square":
            return 1
        elif shape == "circle":
            return 2

    def idx_to_shape(self, idx):
        if idx == 0:
            return "triangle"
        elif idx == 1:
            return "square"
        elif idx == 2:
            return "circle"
