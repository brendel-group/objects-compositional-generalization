from typing import Dict, Tuple, Union
import math
import torch
import torch.nn as nn
import dataclasses

from .config import Config
from . import sampling_utils


def sample_latents(
    n_samples: int,
    n_slots: int,
    cfg: Config,
    sample_mode: str = "random",
    correlation: float = 0,
    delta: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_latents = cfg.get_total_latent_dim

    if sample_mode == "random":
        z = sampling_utils.__sample_random(cfg, n_samples, n_slots, n_latents)

    return z
