import torch
import torch.nn as nn
import numpy as np
from typing import List


class SlotMLP(torch.nn.Module):
    """SlotMLP is based on Vanilla VAE encoder. It takes in an image and outputs a latent vector for each slot."""

    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
        hidden_dims: List = None,
    ) -> None:
        super(SlotMLP, self).__init__()

        self.model_type = "slots_regressor"

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.slot_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[-1] * 4, 256),
                    nn.ELU(),
                    nn.Linear(256, n_slot_latents),
                    nn.Sigmoid()
                )
                for _ in range(n_slots)
            ]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        slot_latents = torch.stack([slot_head(x) for slot_head in self.slot_heads])
        return slot_latents.permute(1, 0, 2)


class FlattenedSlotMLP(torch.nn.Module):
    """FlattenedSlotMLP is based on Vanilla VAE encoder. It takes in an image and outputs a flattened latent vector
    for all slots."""

    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
        hidden_dims: List = None,
    ) -> None:
        super(FlattenedSlotMLP, self).__init__()

        self.model_type = "latents_regressor"

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 4, 256),
            nn.ELU(),
            nn.Linear(256, n_slots * n_slot_latents),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.latent_head(x)
        return x
