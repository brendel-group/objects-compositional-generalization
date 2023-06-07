"""
This file contains functions and wrappers for loading/creating SpritesWorldDataset.
The sole purpose of this file is to make train.py more readable.
"""
import os

import torch
import tqdm

data_path = "/mnt/qb/work/bethge/apanfilov27/object_centric_consistency_project"
code_path = "/mnt/qb/work/bethge/apanfilov27/code/object_centric_ood"


def dump_generated_dataset(dataset: torch.utils.data.TensorDataset, path: str):
    """Dumps generated dataset as torch tensors to a directory."""
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    os.makedirs(os.path.join(path, "latents"), exist_ok=True)

    # taking one sample to get the shape
    image, latent = dataset[0]
    images = torch.Tensor(len(dataset), *image.shape)
    latents = torch.Tensor(len(dataset), *latent.shape)
    for i, (image, latent) in tqdm.tqdm(enumerate(dataset)):
        images[i] = image
        latents[i] = latent
    torch.save(images, os.path.join(path, "images", "images.pt"))
    torch.save(latents, os.path.join(path, "latents", "latents.pt"))


class PreGeneratedDataset(torch.utils.data.Dataset):
    """Loads pre-generated SpriteWorldDataset from a directory."""

    def __init__(self, path: str, n_samples: int = None):
        self.path = path
        self.n_samples = n_samples
        self.images = torch.load(os.path.join(path, "images", "images.pt"))
        self.latents = torch.load(os.path.join(path, "latents", "latents.pt"))

        if self.n_samples is not None:
            self.images = self.images[: self.n_samples]
            self.latents = self.latents[: self.n_samples]
            # print("Truncated dataset to {} samples".format(self.n_samples))
            # #
            # # # 99 - 1 option
            # #
            # # get parent directory of path
            # parent_dir = os.path.dirname(path)
            # print(parent_dir)
            # images_1 = torch.load(
            #     os.path.join(parent_dir, "random", "images", "images.pt")
            # )[: int(self.n_samples * 0.01)]
            # latents_1 = torch.load(
            #     os.path.join(parent_dir, "random", "latents", "latents.pt")
            # )[: int(self.n_samples * 0.01)]
            #
            # # concatenate the two tensors
            # self.images = torch.cat((self.images, images_1), 0)
            # self.latents = torch.cat((self.latents, latents_1), 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.latents[idx]


def collate_fn_normalizer(batch, bias=0, scale=1):
    """Normalize latents target to [0, 1]. Used in dataloader."""
    images, latents = zip(*batch)
    latents = torch.stack(latents)
    latents = (latents - bias) / scale
    return torch.stack(images), latents
