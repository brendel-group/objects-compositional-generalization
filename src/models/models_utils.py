import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def get_encoder(in_channels, out_dim):
    """
    Encoder f^{-1}: X -> Z; X - input image, Z - latent space.
    """
    encoder = nn.Sequential(
        nn.Conv2d(in_channels, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        View((-1, 32 * 4 * 4)),
        nn.Linear(32 * 4 * 4, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, out_dim),
    )
    return encoder


def get_twin_head_encoder(in_channels, out_dim, n_slots):
    """
    Encoder f^{-1}: X -> Z; X - input image, Z - latent space.
    """
    encoder_shared = nn.Sequential()
    encoder_separate = nn.ModuleList(
        [
            nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),
                nn.ELU(),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ELU(),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ELU(),
                nn.Conv2d(32, 32, 4, 2, 1),
                nn.ELU(),
                View((-1, 32 * 4 * 4)),
                nn.Linear(32 * 4 * 4, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, out_dim),
            )
            for _ in range(n_slots)
        ]
    )
    return encoder_shared, encoder_separate


def get_decoder(in_dim, out_channels):
    """
    Decoder f: Z -> X; X - input image, Z - latent space.
    """
    decoder = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 32 * 4 * 4),
        nn.ELU(),
        View((-1, 32, 4, 4)),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
    )
    return decoder
