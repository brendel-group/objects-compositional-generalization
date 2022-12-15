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

    def get_ranges(self) -> Dict[str, Range]:
        result = {}
        for field_name in fields(self):
            if field_name.metadata.get("rv_type") != "categorical":
                result[field_name.name] = Range(
                    min=self[field_name.name].min, max=self[field_name.name].max
                )
            elif field_name.metadata.get("rv_type") == "categorical":
                result[field_name.name] = Range(
                    min=0, max=len(self[field_name.name]) - 1
                )
        return result


@dataclasses.dataclass
class SpriteWorldConfig(Config):
    """
    Config class for SpriteWorld dataset.
    """

    x: Range = field(
        default=Range(0.1, 0.9), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    y: Range = field(
        default=Range(0.2, 0.8), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    shape: List[str] = field(
        default_factory=lambda: [
            "triangle",
            "square",
            # "circle",
            # "pentagon",
            # "hexagon",
            # "octagon", # looks like a circle, when scale is to small
            # "star_4",
            # "star_5",
            # "star_6",
            # "spoke_3",
            # "spoke_4",
            # "spoke_5",
            # "spoke_6",
        ],
        metadata={"rv_type": "categorical", "latent_size": 1},
    )
    scale: Range = field(
        default=Range(0.09, 0.22), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    angle: Range = field(
        default=Range(0, 0), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    c0: Range = field(
        default=Range(0.1, 0.9), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    c1: Range = field(
        default=Range(0.1, 0.9), metadata={"rv_type": "continuous", "latent_size": 1}
    )
    c2: Range = field(
        default=Range(0.1, 0.9), metadata={"rv_type": "continuous", "latent_size": 1}
    )
