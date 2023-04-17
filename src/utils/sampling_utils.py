import numpy as np
import torch

from src.config import Config


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
        l_type = latents_metadata[latent]
        if l_type == "continuous":
            z = cfg[latent].min + (cfg[latent].max - cfg[latent].min) * torch.rand(
                n_samples, n_slots
            )
        elif l_type == "discrete":
            z = torch.randint(cfg[latent].min, cfg[latent].max, (n_samples, n_slots))
        elif l_type == "categorical":
            z = np.random.choice(
                [i for i, category in enumerate(cfg[latent])],
                size=(n_samples, n_slots),
            )
            z = torch.from_numpy(z)
        else:
            raise ValueError(f"Latent type {l_type} not supported.")

        z_out[:, :, i] = z
        i += 1
    return z_out


def sample_delta_pure_off_diagonal_cube(
    n_samples: int, n_slots: int, n_latents: int, delta: float, oversampling: int = 100
) -> torch.Tensor:
    """
    WARNING: Thaddaeus said that this method is redundant, thus this function is not properly tested.

    Sample points where NO component lies within a n_slots-cube with side length delta from the diagonal.

    Function took from by Thaddaeus Wiedemer's code with almost no changes.
    """
    _n = oversampling * n_samples
    z_out = torch.Tensor(0, n_slots, n_latents)
    while z_out.shape[0] < n_samples:
        z_sampled = torch.rand(_n, n_slots, n_latents)
        triuidx = torch.triu_indices(n_slots, n_slots)
        mutual_distances = (
            z_sampled.unsqueeze(1).repeat(1, z_sampled.shape[1], 1, 1)
            - z_sampled.unsqueeze(1).repeat(1, z_sampled.shape[1], 1, 1).transpose(1, 2)
        )[:, triuidx[0], triuidx[1], :]
        mask = (mutual_distances.abs() > delta).any(dim=1).all(dim=1)
        idx = mask.nonzero().squeeze()
        z_out = torch.cat([z_out, z_sampled[idx]])

    z_out = z_out[:n_samples]
    return z_out[:n_samples]


def sample_delta_off_diagonal_cube(
    n_samples: int, n_slots: int, n_latents: int, delta: float, oversampling: int = 100
) -> torch.Tensor:
    """
    All distances from the diagonal are bigger than delta.
    Opposite case of __sample_delta_diagonal_cube.

    Rejection sampling used as the algorithm.
    """
    _n = oversampling * n_samples
    z_out = torch.Tensor(0, n_slots, n_latents)
    while z_out.shape[0] < n_samples:
        # sample uniformly in cube
        z_sampled = torch.rand(_n, n_slots, n_latents)

        diag = torch.ones(_n, n_slots, n_latents)

        # getting orthogonal vectors to the diagonal
        ort_vec = z_sampled - diag * (z_sampled * diag).sum(axis=1, keepdim=True) / (
            diag * diag
        ).sum(axis=1, keepdim=True)

        # get rid of vector if distance to the diagonal is too small
        off_d_mask = (ort_vec.norm(dim=1) > delta).flatten(1).all(1)
        z_sampled = z_sampled[off_d_mask]

        z_out = torch.cat([z_out, z_sampled])

    z_out = z_out[:n_samples]
    return z_out[:n_samples]


def sample_delta_diagonal_cube(
    n_samples: int, n_slots: int, n_latents: int, delta: float, oversampling: int = 100
) -> torch.Tensor:
    """
    Sample near the diagonal in latent space i.e. all distances from the diagonal are less than delta.

    Algorithm:
        1. Draw points on the diagonal of [0, 1)^(n_slots, n_latents) cube.
        2. For every latent draw uniformly noise from n_slots-dimensional ball. For drawing uniformly inside the ball we
            use the following theorem (http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf):
            if point uniformly sampled from the (n+1)-sphere, then n-first coordinates are uniformly sampled from the n-ball.
        3. Project sampled inside-ball points to the hyperplane perpendicular to the diagonal and normalize them
            (this gives us points on (n_slots-2)-sphere embedded in n_slots-space).
        4. Get final points by adding the diagonal point to the projected points.
        5. Keep only points inside the [0, 1)^(n_slots, n_latents) cube.
    """
    _n = oversampling * n_samples
    z_out = torch.Tensor(0, n_slots, n_latents)
    while z_out.shape[0] < n_samples:
        # sample randomly on diagonal
        z_sampled = torch.repeat_interleave(
            torch.rand(_n, n_latents), n_slots, dim=0
        ).reshape(_n, n_slots, n_latents)

        # sample noise from n_slots-ball
        noise = torch.randn(_n, n_slots + 2, n_latents)
        noise = noise / torch.norm(noise, dim=1, keepdim=True)  # points on n-sphere
        noise = noise[:, :n_slots, :]  # remove two last points

        # project to hyperplane perpendicular to diagonal
        ort_vec = noise - z_sampled * (noise * z_sampled).sum(axis=1, keepdim=True) / (
            z_sampled * z_sampled
        ).sum(axis=1, keepdim=True)
        ort_vec /= torch.norm(ort_vec, p=2, dim=1, keepdim=True)

        # final step
        # why n - 1 here? because we sample
        # "radius" not in the original space, but in the embedded
        final = z_sampled + (
            ort_vec
            * torch.pow(torch.rand([_n, 1, n_latents]), 1 / (n_slots - 1))
            * delta
        )

        # only keep samples inside [0, 1]^{k×l}
        mask = ((final - 0.5).abs() <= 0.5).flatten(1).all(1)
        idx = mask.nonzero().squeeze(1)

        z_out = torch.cat([z_out, final[idx]])
    z_out = z_out[:n_samples]
    return z_out[:n_samples]


def sample_diagonal(
    cfg: Config,
    n_samples: int,
    n_slots: int,
    n_latents: int,
    delta: float,
    *,
    mode: str,
) -> torch.Tensor:
    """
    Sample near the diagonal in latent space.

    Args:
        cfg: Config object.
        n_samples: Number of samples.
        n_slots: Number of slots (objects).
        n_latents: Total number of latents.
        delta: Distance from the diagonal [0, 1].
        mode: 'diagonal' or 'off_diagonal'.
    Returns:
        z: Tensor of shape (n_samples, n_slots, n_latents).
    """
    if mode == "diagonal":
        z_out = sample_delta_diagonal_cube(n_samples, n_slots, n_latents, delta)
    elif mode == "off_diagonal":
        z_out = sample_delta_off_diagonal_cube(n_samples, n_slots, n_latents, delta)
    elif mode == "pure_off_diagonal":
        z_out = sample_delta_pure_off_diagonal_cube(
            n_samples, n_slots, n_latents, delta
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    latents_metadata = cfg.get_latents_metadata()

    assert torch.max(z_out) <= 1
    assert torch.min(z_out) >= 0

    i = 0
    for latent in latents_metadata:
        l_type = latents_metadata[latent]
        if l_type == "continuous":
            z_out[:, :, i] = (
                cfg[latent].min + (cfg[latent].max - cfg[latent].min) * z_out[:, :, i]
            )
        elif l_type == "discrete":
            z_out[:, :, i] = torch.round(
                cfg[latent].min + (cfg[latent].max - cfg[latent].min) * z_out[:, :, i]
            )
        elif l_type == "categorical":
            z_out[:, :, i] = torch.floor(len(cfg[latent]) * z_out[:, :, i])
        else:
            raise ValueError(f"Latent type {l_type} not supported.")
        i += 1
    return z_out
