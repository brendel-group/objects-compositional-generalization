import os

import torch
import tqdm


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

    def __init__(self, path: str):
        self.path = path
        self.images = torch.load(os.path.join(path, "images", "images.pt"))
        self.latents = torch.load(os.path.join(path, "latents", "latents.pt"))

        if self.n_samples is not None:
            self.images = self.images
            self.latents = self.latents

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.latents[idx]


class MixedDataset(torch.utils.data.Dataset):
    """
    Loads pre-generated mixed SpriteWorldDataset (with a varying number of objects in the scene) from a directory.
    """

    def __init__(self, path: str, n_samples: int = None):
        self.path = path
        self.n_samples = n_samples

        self.images = self._load_and_concat_images()
        self.latents = self._load_and_concat_latents()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.latents[idx]

    def _load_and_concat_images(self):
        # Get the list of image files
        image_files = sorted(os.listdir(os.path.join(self.path, "images")))
        image_tensors = []

        for image_file in image_files:
            # Load the image tensor
            image_tensor = torch.load(os.path.join(self.path, "images", image_file))

            # Create a tensor of zeros with the desired final shape
            zeros = torch.zeros(
                image_tensor.shape[0],
                5,
                image_tensor.shape[2],
                image_tensor.shape[3],
                image_tensor.shape[4],
            )

            # Fill in the parts of the zeros tensor with the loaded tensor
            zeros[:, -image_tensor.shape[1] :, ...] = image_tensor

            # Add the tensor to the list
            image_tensors.append(zeros)

        # Concatenate all tensors along the first dimension
        return torch.cat(image_tensors, dim=0)

    def _load_and_concat_latents(self):
        # Get the list of latent files
        latent_files = sorted(os.listdir(os.path.join(self.path, "latents")))
        latent_tensors = []

        for latent_file in latent_files:
            # Load the latent tensor
            latent_tensor = torch.load(os.path.join(self.path, "latents", latent_file))

            # Create a tensor of zeros with the desired final shape
            zeros = torch.zeros(latent_tensor.shape[0], 4, latent_tensor.shape[2])

            # Fill in the parts of the zeros tensor with the loaded tensor
            zeros[:, : latent_tensor.shape[1], :] = latent_tensor

            # Add the tensor to the list
            latent_tensors.append(zeros)

        # Concatenate all tensors along the first dimension
        return torch.cat(latent_tensors, dim=0)


def collate_fn_normalizer(batch, bias=0, scale=1, mixed=False, device="cpu"):
    """Normalize latents target to [0, 1]. Used in dataloader."""
    images, latents = zip(*batch)
    latents = torch.stack(latents)
    if mixed:
        return torch.stack(images), latents
    latents = (latents - bias) / scale
    return torch.stack(images).to(device), latents.to(device)


def filter_objects(latents, max_samples=5000, threshold=0.2, sort=False):
    """
    Filter objects based on their Euclidean distance.
    Args:
        latents: Tensor of shape (batch_size, n_slots, n_latents)
        max_objects: Number of objects to keep at most
        threshold: Distance threshold
        sort: Whether to sort the objects by distance
    """
    N, slots, _ = latents.size()
    mask = torch.zeros(N, dtype=bool)

    # Compute Euclidean distance for each pair of slots in each item
    for n in range(N):
        slots_distances = torch.cdist(latents[n, :, :2], latents[n, :, :2], p=2)
        slots_distances.fill_diagonal_(float("inf"))  # Ignore distance to self

        # Only keep samples in which no two objects are closer than the threshold
        min_distance = slots_distances.min().item()
        if min_distance >= threshold:
            mask[n] = True

    # If all objects are "close", print a message and return
    if not torch.any(mask):
        print("No objects were found that meet the distance threshold.")
        return None, []

    # Apply the mask to the latents
    filtered_samples = latents[mask]
    filtered_indices = torch.arange(N)[mask]

    # If the number of filtered samples exceeds the maximum, truncate them
    if filtered_samples.size(0) > max_samples:
        filtered_samples = filtered_samples[:max_samples]
        filtered_indices = filtered_indices[:max_samples]

    return filtered_samples, filtered_indices.tolist()
