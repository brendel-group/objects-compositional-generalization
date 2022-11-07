import torch
import numpy as np

from .config import Config


def __sample_random(
    cfg: Config, n_samples: int, n_slots: int, n_latents: int
) -> torch.Tensor:
    """
    Sample randomly in complete latent space.

    Args:
        cfg: Config object.
        n_samples: Number of samples.
        n_slots: Number of slots (objects).
        n_latents: Total number of latents.

    Returns:
        z: Tensor of shape (n_samples, n_slots, n_latents).
    """
    z_out = torch.empty(n_samples, n_slots, n_latents)
    latents_metadata = cfg.get_latents_metadata()
    i = 0
    for latent in latents_metadata:
        l_type, l_size = latents_metadata[latent]
        if l_type == "continuous":
            z = cfg[latent].min + (cfg[latent].max - cfg[latent].min) * torch.rand(
                n_samples, n_slots, l_size
            )
        elif l_type == "discrete":
            z = torch.randint(
                cfg[latent].min, cfg[latent].max, (n_samples, n_slots, l_size)
            )
        elif l_type == "categorical":
            z = np.random.choice(
                [i for i, category in enumerate(cfg[latent])],
                size=(n_samples, n_slots, l_size),
            )
            z = torch.from_numpy(z)
        else:
            raise ValueError(f"Latent type {l_type} not supported.")

        z_out[:, :, i : i + l_size] = z
        i += l_size
    return z_out


def __sample_diagonal(
    cfg: Config, n_samples: int, n_slots: int, n_latents: int, delta: float
) -> torch.Tensor:
    """
    Sample near the diagonal in latent space.

    Args:
        cfg: Config object.
        n_samples: Number of samples.
        n_slots: Number of slots (objects).
        n_latents: Total number of latents.
        delta: Distance from the diagonal [0, 1].

    Returns:
        z: Tensor of shape (n_samples, n_slots, n_latents).
    """
    z_diag = torch.repeat_interleave(
        torch.rand(n_samples, n_latents), n_slots, dim=0
    ).reshape(n_samples, n_slots, n_latents)
    z_out = z_diag + (torch.rand(n_samples, n_slots, n_latents) - 0.5) * delta * 2

    z_out = torch.where(z_out > 1, 2 - z_out, z_out)
    z_out = torch.where(z_out < 0, z_out.abs(), z_out)
    latents_metadata = cfg.get_latents_metadata()

    assert torch.max(z_out) <= 1
    assert torch.min(z_out) >= 0

    i = 0
    for latent in latents_metadata:
        l_type, l_size = latents_metadata[latent]
        if l_type == "continuous":
            z_out[:, :, i : i + l_size] = (
                cfg[latent].min
                + (cfg[latent].max - cfg[latent].min) * z_out[:, :, i : i + l_size]
            )
        elif l_type == "discrete":
            z_out[:, :, i : i + l_size] = torch.round(
                cfg[latent].min
                + (cfg[latent].max - cfg[latent].min) * z_out[:, :, i : i + l_size]
            )
        elif l_type == "categorical":
            z_out[:, :, i : i + l_size] = torch.floor(
                len(cfg[latent]) * z_diag[:, :, i : i + l_size]
            )
            mask = np.random.choice(
                [0, 1], size=(n_samples, n_slots, l_size), p=[1 - delta, delta]
            )  # taking same category with probability max((1-delta) + 1 / n_categories, 1)
            mask = torch.from_numpy(mask)
            z_out[:, :, i : i + l_size] = torch.where(
                mask == 1,
                torch.randint(0, len(cfg[latent]), (n_samples, n_slots, l_size)),
                z_out[:, :, i : i + l_size],
            )
        else:
            raise ValueError(f"Latent type {l_type} not supported.")
        i += l_size
    return z_out
