import torch
import torch.nn as nn
import numpy as np
from typing import List


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def get_encoder(in_channels, out_dim):
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
        nn.Linear(256, out_dim),
    )
    return encoder


def get_decoder(in_dim, out_channels):
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


class SlotMLPMonolithic(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPMonolithic, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = get_encoder(in_channels, n_slots * n_slot_latents)
        self.decoder = get_decoder(n_slots * n_slot_latents, in_channels)

    def forward(self, x):
        latents = self.encoder(x)
        image = self.decoder(latents)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)
        return image, latents


class SlotMLPAdditive(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPAdditive, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = get_encoder(in_channels, n_slots * n_slot_latents)
        self.decoder = get_decoder(n_slot_latents, in_channels)

    def forward(self, x):
        latents = self.encoder(x)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)

        image = 0
        for i in range(self.n_slots):
            image += self.decoder(latents[:, i, :])

        return image, latents


class SlotMLPEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = get_encoder(in_channels, n_slots * n_slot_latents)

    def forward(self, x):
        latents = self.encoder(x)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)
        return latents
