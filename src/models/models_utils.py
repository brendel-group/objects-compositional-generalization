import torch.nn as nn
import torch

import torch.nn.functional as F


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


def add_figures_with_obstruction(
    figure_1: torch.Tensor, figure_2: torch.Tensor
) -> torch.Tensor:
    """Add two figures with pixels of figure_1 overlapping figure_2."""
    mask_1 = figure_1 > 0
    mask_2 = figure_2 > 0
    mask = mask_1 * mask_2
    return figure_1 + figure_2 * (~mask)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, nonlinear=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = [hidden_dim] * (self.n_layers - 1)

        if nonlinear:
            self._act_f = lambda x: F.leaky_relu(x, negative_slope=0.2)
        else:
            self._act_f = lambda x: x

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim)
            )
        self.fc = nn.ModuleList(_fc_list)

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f(self.fc[c](h))
        return h
