import warnings

import numpy as np
import torch
import tqdm
from spriteworld import environment, renderers, sprite, tasks

from . import sampling_utils
from .config import Config, SpriteWorldConfig

warnings.filterwarnings("ignore", module="spriteworld")


def sample_latents(
    n_samples: int,
    n_slots: int,
    cfg: Config,
    sample_mode: str = "random",
    delta: float = 1,
) -> torch.Tensor:

    assert 0 < delta <= 1, "Delta must be in (0, 1]."

    n_latents = cfg.get_total_latent_dim
    if sample_mode == "random":
        z = sampling_utils.sample_random(cfg, n_samples, n_slots, n_latents)
    elif sample_mode == "diagonal":
        z = sampling_utils.sample_diagonal(
            cfg, n_samples, n_slots, n_latents, delta, mode="diagonal"
        )
    elif sample_mode == "off_diagonal":
        z = sampling_utils.sample_diagonal(
            cfg, n_samples, n_slots, n_latents, delta, mode="off_diagonal"
        )
    elif sample_mode == "pure_off_diagonal":
        z = sampling_utils.sample_diagonal(
            cfg, n_samples, n_slots, n_latents, delta, mode="pure_off_diagonal"
        )
    else:
        raise ValueError(f"Sample mode {sample_mode} not supported.")
    return z


class SpriteWorldDataset(torch.utils.data.TensorDataset):
    """
    Class for generating sprites from Sprite-World environment.

    Args:
        n_samples: Number of samples to generate.
        n_slots: Number of objects in scene.
        cfg: Instance of SpriteWorldConfig class, determines the range of values for each latent variable.
        sample_mode: Sampling mode for latent variables ("random", "diagonal", "off_diagonal", "pure_off_diagonal").
        img_h: Height of the generated images.
        img_w: Width of the generated images.
        delta: Delta for "diagonal", "off_diagonal" and "pure_off_diagonal" sampling. Should be in (0, 1].
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

        if self.no_overlap:
            # here we decrease delta to the range where it is possible to have no overlapping figures analytically it
            # is enough to have max_delta = (1 / (n_slots)) * 0.5 to have not intersecting x-coordinates,
            # but we decrease it a bit more due to the fact that figures also have some width
            max_delta = (1 / (self.n_slots + 2)) * 0.5
            if self.delta > max_delta:
                print(
                    f"Delta is too big for 'no_overlap' mode, setting it to {max_delta}."
                )
                self.delta = max_delta
        self.z = sample_latents(
            n_samples, n_slots, cfg, sample_mode, delta=self.delta, **kwargs
        )

        self.__generate_ind = 0
        self.env = environment.Environment(
            task=tasks.NoReward(),
            action_space=None,
            renderers=self.renderer_config,
            init_sprites=self.__generate,
            max_episode_length=1,
        )

        self.x = self.__generate_from_latents().detach()
        super().__init__(self.x, self.z)

    def __generate_from_latents(self) -> torch.Tensor:
        images = [None] * self.n_samples
        for sample_ind in tqdm.tqdm(
            range(self.n_samples),
            desc=f"Generating images (sampling: {self.sample_mode})",
        ):
            self.__generate_ind = sample_ind
            ts = self.env.reset()
            images[sample_ind] = torch.from_numpy(
                np.array(ts.observation["image"])[-1]
            )  # last one contains all sprites in one scene
        self.__generate_ind = 0
        return torch.stack(images, dim=0)

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
                ort_vec *= torch.pow(torch.rand(1), 1 / (self.n_slots - 1)) * self.delta
                x_diag += ort_vec

            elif self.sample_mode == "off_diagonal":
                _n = 10
                z_out = torch.Tensor(0, self.n_slots, 1)
                while z_out.shape[0] < _n:
                    z_sampled = torch.rand(_n, self.n_slots, 1)
                    diag = torch.ones(_n, self.n_slots, 1)
                    ort_vec = z_sampled - diag * (z_sampled * diag).sum(
                        axis=1, keepdim=True
                    ) / (diag * diag).sum(axis=1, keepdim=True)
                    off_d_mask = (ort_vec.norm(dim=1) > self.delta).flatten(1).all(1)
                    z_sampled = z_sampled[off_d_mask]

                    z_out = torch.cat([z_out, z_sampled])
                x_diag = z_out[:1].squeeze()

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

    def __generate(self):
        """Generates a list of sprites from generated latents for the environment."""

        if self.sample_mode in ["diagonal", "off_diagonal", "pure_off_diagonal"]:
            # adjusting x to avoid overlapping sprites
            x_scaled = self.__adjust_x_coord(self.z[self.__generate_ind, :, :2])
            self.z[self.__generate_ind, :, 0] = x_scaled

        if self.sample_mode in ["off_diagonal", "pure_off_diagonal"]:
            # removing same shapes (this artifact comes from "floor" rounding)
            shape = self.__adjust_shape(self.z[self.__generate_ind, :, 2])
            self.z[self.__generate_ind, :, 2] = shape

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
                    i += 1
                else:
                    l_type, l_size = latents_metadata[latent]
                    sprite_sample[latent] = sample[slot_ind, i : i + l_size]
                    i += l_size
            sampled_sprites[slot_ind] = sprite.Sprite(**sprite_sample)
        return sampled_sprites
