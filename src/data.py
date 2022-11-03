from typing import Dict, Tuple, Union
import math
import torch
import torch.nn as nn
import dataclasses
import numpy as np

from spriteworld import sprite
from spriteworld import renderers
from spriteworld import tasks
from spriteworld import environment

from .config import Config, SpriteWorldConfig
from . import sampling_utils


def sample_latents(
    n_samples: int,
    n_slots: int,
    cfg: Config,
    sample_mode: str = "random",
    correlation: float = 0,
    delta: float = 0,
) -> torch.Tensor:
    n_latents = cfg.get_total_latent_dim

    if sample_mode == "random":
        z = sampling_utils.__sample_random(cfg, n_samples, n_slots, n_latents)
    return z


class SpriteWorldDataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        n_samples: int,
        n_slots: int,
        cfg: SpriteWorldConfig,
        sample_mode: str = "random",
        img_h: int = 64,
        img_w: int = 64,
    ):
        self.n_samples = n_samples
        self.n_slots = n_slots
        self.cfg = cfg
        self.img_h = img_h
        self.img_w = img_w

        self.renderer_config = {
            "image": renderers.PILRenderer(
                image_size=(self.img_h, self.img_w),
                anti_aliasing=5,
                color_to_rgb=renderers.color_maps.hsv_to_rgb,
            ),
            "attributes": renderers.SpriteFactors(
                factors=("x", "y", "shape", "angle", "scale", "c0", "c1", "c2")
            ),
        }
        self.z = sample_latents(n_samples, n_slots, cfg, sample_mode)
        self.__generate_ind = 0

        self.env = environment.Environment(
            task=tasks.NoReward(),
            action_space=None,
            renderers=self.renderer_config,
            init_sprites=self.__generate,
            max_episode_length=1,
        )

        self.x = self.generate_from_latents().detach()
        super().__init__(self.x, self.z)

    def generate_from_latents(self) -> torch.Tensor:
        images = [None] * self.n_samples
        for sample_ind in range(self.n_samples):
            self.__generate_ind = sample_ind
            ts = self.env.reset()
            images[sample_ind] = torch.from_numpy(
                np.array(ts.observation["image"])[-1]
            )  # last one contains all sprites in on scene
        self.__generate_ind = 0
        return torch.stack(images, dim=0)

    def __generate(self):
        sample = self.z[self.__generate_ind]
        latents_metadata = self.cfg.get_latents_metadata()
        sampled_sprites = [None] * self.n_slots
        for slot_ind in range(self.n_slots):
            sprite_sample = {}
            i = 0
            for latent in latents_metadata:
                if latent == "shape":
                    sprite_sample["shape"] = self.cfg.shape[
                        int(sample[slot_ind, i].item())
                    ]
                    i += 1
                else:
                    l_type, l_size = latents_metadata[latent]
                    sprite_sample[latent] = sample[slot_ind, i : i + l_size]
                    i += l_size
            sampled_sprites[slot_ind] = sprite.Sprite(**sprite_sample)
        return sampled_sprites
