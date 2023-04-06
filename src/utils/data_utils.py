import os

import torch
import tqdm

import src.utils.training_utils as training_utils

def dump_generated_dataset(dataset: torch.utils.data.TensorDataset, path: str):
    """Dumps generated dataset as torch tensors to a directory."""
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    os.makedirs(os.path.join(path, "latents"), exist_ok=True)

    images = torch.Tensor(len(dataset), 3, 64, 64)
    latents = torch.Tensor(len(dataset), 2, 5)
    for i, (image, latent) in tqdm.tqdm(enumerate(dataset)):
        images[i] = image
        latents[i] = latent
    torch.save(images, os.path.join(path, "images", "images.pt"))
    torch.save(latents, os.path.join(path, "latents", "latents.pt"))


class PreGeneratedDataset(torch.utils.data.Dataset):
    """Loads pre-generated SpriteWorldDataset from a directory."""

    def __init__(self, path: str):
        self.path = path
        self.images = torch.load(os.path.join(path, "images", "images.pt"))
        self.latents = torch.load(os.path.join(path, "latents", "latents.pt"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.latents[idx]


def load_identifiability_dataset(path: str, min_offset, scale):
    """Loads identifiability dataset from a directory."""

    dataset = PreGeneratedDataset(path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda b: training_utils.collate_fn_normalizer(b, min_offset, scale),
    )
    print(f"Identifiability dataset successfully loaded from {path}.")
    return dataloader
