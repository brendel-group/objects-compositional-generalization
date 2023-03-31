import numpy as np
import torch


def collate_fn_normalizer(batch, bias=0, scale=1):
    """Normalize latents target to [0, 1]. Used in dataloader."""
    images, latents = zip(*batch)
    latents = torch.stack(latents)
    latents = (latents - bias) / scale
    return torch.stack(images), latents


def sample_z_from_latents(latents):
    """
    Sample "delusional" z samples from latents.

    Args:
        latents: tensor of shape (batch_size, n_slots, n_latents)

    Returns:
        sampled_z: tensor of shape (batch_size, n_slots, n_latents)
    """
    batch_size, n_slots, n_latents = latents.shape

    # Flatten the latents tensor into a 2D tensor
    flattened_latents = latents.view(batch_size * n_slots, n_latents)

    # Sample indices randomly with replacement from the flattened latents tensor
    indices = np.random.choice(
        len(flattened_latents),
        size=batch_size * n_slots,
    )

    # Gather the sampled latents from the flattened tensor
    sampled_latents = flattened_latents[indices]

    # Reshape the sampled latents tensor back to the original shape
    sampled_z = sampled_latents.view(batch_size, n_slots, n_latents)

    return sampled_z.to(latents.device)


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
