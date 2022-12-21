import torch
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
        nn.Linear(256, out_dim),
    )
    return encoder


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
        nn.Dropout2d(0.3),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.Dropout2d(0.3),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
    )
    return decoder


class SlotMLPMonolithic(torch.nn.Module):
    """
    Models predicts latent vector z_hat = f^{-1}(x), where z_hat = [z_hat_1, z_hat_2, ..., z_hat_n_slots] and
    generates x_hat = f(z_hat). Model outputs x_hat and z_hat reshaped into (batch_size, n_slots, n_slot_latents).
    """

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
        self.model_name = "SlotMLPMonolithic"

    def forward(self, x):
        latents = self.encoder(x)
        image = self.decoder(latents)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)
        return image, latents


class SlotMLPAdditive(torch.nn.Module):
    """
    Model predicts latent vector z_hat_i = f^{-1}(x) for each slot and generates x_hat = sum_{i=1}^{n_slots} f(z_hat_i),
    by summing the output of each slot. For the output z_hat reshaped into (batch_size, n_slots, n_slot_latents).
    Model outputs x_hat, z_hat and list of x_hat_i.
    """

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
        self.model_name = "SlotMLPAdditive"

    def forward(self, x):
        latents = self.encoder(x)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)

        image = 0
        figures = []
        for i in range(self.n_slots):
            figure = self.decoder(latents[:, i, :])
            image += figure
            figures.append(figure)
        return image, latents, figures


class SlotMLPEncoder(torch.nn.Module):
    """
    Models predicts whole latent vector z_hat = f^{-1}(x), where z_hat = [z_hat_1, z_hat_2, ..., z_hat_n_slots] and
    generates x_hat = f(z_hat). Model outputs z_hat reshaped into (batch_size, n_slots, n_slot_latents).
    """

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
        self.model_name = "SlotMLPEncoder"

    def forward(self, x):
        latents = self.encoder(x)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)
        return latents


class SlotMLPAdditiveDecoder(torch.nn.Module):
    """
    Model generates x_hat = sum_{i=1}^{n_slots} f(z_i),
    by summing the output of each slot.  Model outputs x_hat and list of x_hat_i.
    """

    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPAdditiveDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.decoder = get_decoder(n_slot_latents, in_channels)
        self.model_name = "SlotMLPAdditiveDecoder"

    def forward(self, latents):
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)
        image = 0
        figures = []
        for i in range(self.n_slots):
            figure = self.decoder(latents[:, i, :])
            image += figure
            figures.append(figure)
        return image, figures


class SlotMLPMonolithicDecoder(torch.nn.Module):
    """
    Models generates x_hat = f(z). Model outputs x_hat.
    """

    def __init__(
        self,
        in_channels: int,
        n_slots: int,
        n_slot_latents: int,
    ) -> None:
        super(SlotMLPMonolithicDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.decoder = get_decoder(n_slots * n_slot_latents, in_channels)
        self.model_name = "SlotMLPMonolithicDecoder"

    def forward(self, latents):
        image = self.decoder(latents.view(-1, self.n_slots * self.n_slot_latents))
        return image
