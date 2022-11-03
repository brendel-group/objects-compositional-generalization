import torch

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
            z = torch.multinomial(
                torch.tensor(
                    [i for i, category in enumerate(cfg[latent])], dtype=torch.float32
                ),
                n_samples * n_slots * l_size,
                replacement=True,
            ).reshape(n_samples, n_slots, l_size)
        else:
            raise ValueError(f"Latent type {l_type} not supported.")

        z_out[:, :, i: i + l_size] = z
        i += l_size
    return z_out
