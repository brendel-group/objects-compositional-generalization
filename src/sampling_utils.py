import numpy as np
import torch

from .config import Config


def sample_random(
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


def __sample_delta_diagonal_cube(
    n_samples: int, n_slots: int, n_latents: int, delta: float, oversampling: int = 100
) -> torch.Tensor:
    _n = oversampling * n_samples
    z_out = torch.Tensor(0, n_slots, n_latents)
    while z_out.shape[0] < n_samples:
        # sample randomly on diagonal
        z_sampled = torch.repeat_interleave(
            torch.rand(n_samples, n_latents), n_slots, dim=0
        ).reshape(n_samples, n_slots, n_latents)

        # apply random offset
        eps = torch.rand(n_samples, n_slots, n_latents) * 2 * delta - delta
        z_sampled += eps

        # remove offset from one original sample
        z_sampled[:, 0, :] -= eps[:, 0, :]

        # only keep samples inside [0, 1]^{k×l}
        mask = ((z_sampled - 0.5).abs() <= 0.5).flatten(1).all(1)
        idx = mask.nonzero().squeeze(1)
        z_out = torch.cat([z_out, z_sampled[idx]])

    return z_out[:n_samples]


def sample_diagonal(
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
    z_out = __sample_delta_diagonal_cube(n_samples, n_slots, n_latents, delta)
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
                len(cfg[latent]) * z_out[:, :, i : i + l_size]
            )
        else:
            raise ValueError(f"Latent type {l_type} not supported.")
        i += l_size
    return z_out
