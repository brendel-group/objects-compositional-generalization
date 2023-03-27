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
    flattened_latents = latents.view(-1, latents.shape[-1])
    sampled_z = torch.stack(
        [
            flattened_latents[
                np.random.choice(
                    len(flattened_latents),
                    size=len(latents),
                ),
                :,
            ]
            for _ in range(latents.shape[1])
        ],
        dim=1,
    )
    return sampled_z.to(latents.device)


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
