import os
import warnings
from typing import Callable, Optional

import numpy as np
import torch
import tqdm
from PIL import Image
from spriteworld import environment, renderers, sprite, tasks
from src.datasets.configs import SpriteWorldConfig
from src.utils.sampling_utils import sample_latents
from torchvision import transforms

warnings.filterwarnings("ignore", module="spriteworld")


class SpriteWorldDataset(torch.utils.data.TensorDataset):
    """
    Class for generating sprites from Sprite-World environment.

    Args:
        n_samples: Number of samples to generate.
        n_slots: Number of objects in scene. For n_slots=1, always 'random' sampling is used.
        cfg: Instance of SpriteWorldConfig class, determines the range of values for each latent variable.
        sample_mode: Sampling mode for latent variables ("random", "diagonal", "off_diagonal").
        img_h: Height of the generated images.
        img_w: Width of the generated images.
        delta: Delta for "diagonal", "off_diagonal" and sampling. Should be in (0, 1].
        no_overlap: Whether to allow overlapping of sprites.
            For sample_mode=="diagonal" and "off_diagonal" provided delta decreased to the range where it is possible
            to have no overlapping figures.
    """

    def __init__(
        self,
        n_samples: int,
        n_slots: int,
        cfg: SpriteWorldConfig,
        sample_mode: str = "random",
        img_h: int = 64,
        img_w: int = 64,
        delta: float = 1,
        no_overlap: bool = False,
        transform: Optional[Callable] = None,
        z: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.n_samples = n_samples
        self.n_slots = n_slots
        self.cfg = cfg
        self.img_h = img_h
        self.img_w = img_w
        self.delta = delta
        self.sample_mode = sample_mode
        self.no_overlap = no_overlap
        self.transform = transform

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
        if n_slots == 1:
            self.sample_mode = "random"
            print("For n_slots=1 always 'random' sampling is used.")

        if self.no_overlap and self.sample_mode in ["diagonal", "off_diagonal"]:
            # here we decrease delta to the range where it is possible to have no overlapping figures analytically it
            # is enough to have max_delta = (1 / (n_slots)) * 0.5 to have not intersecting x-coordinates,
            # but we decrease it a bit more due to the fact that figures also have some width
            max_delta = (1 / (self.n_slots + 2)) * 0.5
            if self.delta > max_delta:
                print(
                    f"Delta is too big for 'no_overlap' mode, setting it to {max_delta}."
                )
                self.delta = max_delta
        self.ignore_adjusment = False
        if z is None:
            self.z = sample_latents(
                self.n_samples,
                self.n_slots,
                self.cfg,
                self.sample_mode,
                delta=self.delta,
                **kwargs,
            )
        else:
            self.z = z
            self.ignore_adjusment = True

        self.__generate_ind = 0
        self.env = environment.Environment(
            task=tasks.NoReward(),
            action_space=None,
            renderers=self.renderer_config,
            init_sprites=self.__generate,
            max_episode_length=1,
        )

        self.x = self.__generate_from_latents().detach()
        super().__init__(self.x, self.__update_latents(self.z))

    def __update_latents(self, z: torch.Tensor):
        """Remove fixed latents."""
        return torch.cat([z[:, :, :4], z[:, :, 5:-2]], dim=-1)

    def __generate(self):
        """Generates a list of sprites from generated latents for the environment."""

        if (
            self.sample_mode in ["diagonal", "off_diagonal", "pure_off_diagonal"]
            and not self.ignore_adjusment
        ):
            # adjusting x to avoid overlapping sprites
            x_scaled = self.__adjust_x_coord(self.z[self.__generate_ind, :, :2])
            self.z[self.__generate_ind, :, 0] = x_scaled

        if (
            self.sample_mode in ["off_diagonal", "pure_off_diagonal"]
            and not self.ignore_adjusment
        ):
            # removing same shapes (this artifact comes from "floor" rounding)
            shape = self.__adjust_shape(self.z[self.__generate_ind, :, 2])
            self.z[self.__generate_ind, :, 2] = shape

        if not self.ignore_adjusment:
            # adjusting figure scale to avoid severely overlapping sprites
            self.z[self.__generate_ind, :, 3] = self.cfg["scale"].min + (
                self.z[self.__generate_ind, :, 3] - self.cfg["scale"].min
            ) * (1 / (self.n_slots + 1))

        # generating sprites
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
                else:
                    sprite_sample[latent] = sample[slot_ind, i]
                i += 1
            sampled_sprites[slot_ind] = sprite.Sprite(**sprite_sample)
        return sampled_sprites

    def __generate_from_latents(self) -> torch.Tensor:
        images = torch.zeros(
            (self.n_samples, self.n_slots + 1, 3, self.img_h, self.img_w)
        )
        for sample_ind in tqdm.tqdm(
            range(self.n_samples),
            desc=f"Generating images (sampling: {self.sample_mode})",
        ):
            self.__generate_ind = sample_ind
            ts = self.env.reset()
            out = torch.from_numpy(np.array(ts.observation["image"]))

            images[sample_ind] = out.permute(0, 3, 1, 2)

            if self.transform is not None:
                for slot_ind in range(self.n_slots + 1):
                    images[sample_ind, slot_ind] = self.transform(
                        images[sample_ind, slot_ind].squeeze() * 255
                    )

        self.__generate_ind = 0
        stacked_images = images.to(torch.float32)

        return stacked_images

    def __adjust_x_coord(self, generated_x_y) -> torch.Tensor:
        """
        Generates x coordinate separately to avoid overlapping sprites.
        """

        if self.sample_mode in ["diagonal", "off_diagonal"] and self.n_slots > 1:
            if self.sample_mode == "diagonal":
                x_diag = torch.repeat_interleave(torch.rand(1), self.n_slots)

                noise = torch.randn(self.n_slots + 2)
                noise = noise / torch.norm(noise, keepdim=True)
                noise = noise[: self.n_slots]

                ort_vec = noise - x_diag * torch.dot(noise, x_diag) / torch.dot(
                    x_diag, x_diag
                )
                ort_vec = ort_vec / torch.norm(ort_vec, keepdim=True)
                # why n - 1 here? because we sample "radius" not in the original space, but in the embedded
                ort_vec *= torch.pow(torch.rand(1), 1 / (self.n_slots - 1)) * self.delta
                x_diag += ort_vec

            elif self.sample_mode == "off_diagonal":
                x_diag = generated_x_y[:, 0]

            if self.no_overlap:
                k = 1 / self.n_slots
            else:
                k = ((1 / self.n_slots) + self.delta) / 2

            const = torch.FloatTensor([k * i for i in range(self.n_slots)])
            x_diag += const
            x_diag = x_diag % 1

            x_scaled = (
                self.cfg["x"].min + (self.cfg["x"].max - self.cfg["x"].min) * x_diag
            )
            return x_scaled
        else:
            generated_x = generated_x_y[:, 0]
            return generated_x

    def __adjust_shape(self, generated_shape) -> torch.Tensor:
        """
        Generates shape separately to avoid same figures for off_diagonal and pure_off_diagonal sampling.
        """
        while (
            sum(
                [
                    generated_shape[0].item() == generated_shape[i].item()
                    for i in range(self.n_slots)
                ]
            )
            == self.n_slots
        ):
            generated_shape = torch.rand(self.n_slots)
            generated_shape = torch.floor(len(self.cfg["shape"]) * generated_shape)
        return generated_shape
