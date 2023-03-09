import torch

from . import models_utils

from ..training_utils import (
    sample_z_from_gt,

)


class SlotEncoder(torch.nn.Module):
    def __init__(self, in_channels, n_slots, n_slot_latents):
        super(SlotEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = models_utils.get_encoder(in_channels, n_slots * n_slot_latents)

    def forward(self, x):
        x = self.encoder(x)
        out = x.view(-1, self.n_slots, self.n_slot_latents)
        return out


class TwinHeadedSlotEncoder(torch.nn.Module):
    def __init__(self, in_channels, n_slots, n_slot_latents):
        super(TwinHeadedSlotEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder_shared, self.encoder_separate = models_utils.get_twin_head_encoder(
            in_channels, n_slots, n_slot_latents
        )

    def forward(self, x):
        x = self.encoder_shared(x)
        out = torch.stack([encoder(x) for encoder in self.encoder_separate], dim=1)
        out = out.view(-1, self.n_slots, self.n_slot_latents)
        return out


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
        self.decoder = models_utils.get_decoder(n_slot_latents, in_channels)
        self.model_name = "SlotMLPAdditiveDecoder"

    def forward(self, latents):
        # image = torch.zeros_like(self.decoder(latents[0, 0]))
        image = 0
        figures = []
        for i in range(self.n_slots):
            figure = self.decoder(latents[:, i, :])
            image += figure
            # models_utils.add_figures_with_obstruction(image, figure)
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
        self.decoder = models_utils.get_decoder(n_slots * n_slot_latents, in_channels)
        self.model_name = "SlotMLPMonolithicDecoder"

    def forward(self, latents):
        image = self.decoder(latents.view(-1, self.n_slots * self.n_slot_latents))
        return image


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
        self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents)
        self.decoder = SlotMLPMonolithicDecoder(in_channels, n_slots, n_slot_latents)
        self.model_name = "SlotMLPMonolithic"

    def forward(self, x):
        latents = self.encoder(x)
        latents = latents.view(-1, self.n_slots * self.n_slot_latents)
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
        twin_headed: bool = False,
    ) -> None:
        super(SlotMLPAdditive, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.twin_headed = twin_headed
        if twin_headed:
            self.encoder = TwinHeadedSlotEncoder(in_channels, n_slots, n_slot_latents)
        else:
            self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents)
        self.decoder = SlotMLPAdditiveDecoder(in_channels, n_slots, n_slot_latents)
        self.model_name = "SlotMLPAdditive"

    def forward(self, x, use_consistency_loss=False, detached_latents=False):
        latents = self.encoder(x)

        if detached_latents:
            image, figures = self.decoder(latents.detach())
        else:
            image, figures = self.decoder(latents)

        if use_consistency_loss:
            sampled_z = sample_z_from_gt(latents.detach())
            with torch.no_grad():
                x_hat, figures_hat = self.decoder(sampled_z)

            z_hat = self.encoder(x_hat)
            return image, latents, figures, z_hat, x_hat, figures_hat, sampled_z
        else:
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
        twin_headed: bool = False,
    ) -> None:
        super(SlotMLPEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.twin_headed = twin_headed
        if twin_headed:
            self.encoder = TwinHeadedSlotEncoder(in_channels, n_slots, n_slot_latents)
        else:
            self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents)
        self.model_name = "SlotMLPEncoder"

    def forward(self, x):
        latents = self.encoder(x)
        return latents
