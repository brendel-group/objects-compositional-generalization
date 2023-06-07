from contextlib import nullcontext
from typing import Dict, Any

import torch

from src.utils.training_utils import sample_z_from_latents
from . import utils


class SlotEncoder(torch.nn.Module):
    def __init__(self, in_channels, n_slots, n_slot_latents):
        super(SlotEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = utils.get_encoder(in_channels, n_slots * n_slot_latents)

    def forward(self, x):
        x = self.encoder(x)
        out = x.view(-1, self.n_slots, self.n_slot_latents)
        return out


class TwinHeadedSlotEncoder(torch.nn.Module):
    def __init__(self, in_channels, n_slots, n_slot_latents):
        super(TwinHeadedSlotEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder_shared, self.encoder_separate = utils.get_twin_head_encoder(
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
        self.decoder = utils.get_decoder(n_slot_latents, in_channels)
        self.model_name = "SlotMLPAdditiveDecoder"

    def forward(self, latents):
        # Reshape latents from [batch_size, n_slots, features] to [batch_size*n_slots, features]
        batch_size, n_slots, features = latents.size()
        reshaped_latents = latents.view(batch_size * n_slots, features)

        # Pass reshaped latents through decoder
        reshaped_figures = self.decoder(reshaped_latents)

        # Reshape figures back to [batch_size, n_slots, ...]
        figures = reshaped_figures.view(
            batch_size, n_slots, *reshaped_figures.shape[1:]
        )
        # Sum over the n_slots dimension for image
        image = figures.sum(dim=1)

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
        self.decoder = utils.get_decoder(n_slots * n_slot_latents, in_channels)
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

    def consistency_pass(
        self, hat_z, figures, use_consistency_loss, extended_consistency_loss
    ):
        # getting imaginary samples
        with torch.no_grad():
            z_sampled, indices = sample_z_from_latents(hat_z.detach())
            figures_sampled = figures.reshape(
                -1, figures.shape[2], figures.shape[3], figures.shape[4]
            )[indices].reshape(-1, *figures.shape[1:])
            x_sampled = torch.sum(figures_sampled, dim=1)

        # encoder pass
        with nullcontext() if (
            use_consistency_loss or extended_consistency_loss
        ) else torch.no_grad():
            hat_z_sampled = self.encoder(x_sampled)

        # decoder pass
        with nullcontext() if extended_consistency_loss else torch.no_grad():
            hat_x_sampled, _ = self.decoder(hat_z_sampled)

        return {
            "sampled_image": x_sampled,
            "sampled_figures": figures_sampled.permute(1, 0, 2, 3, 4),
            "sampled_latents": z_sampled,
            "reconstructed_sampled_image": hat_x_sampled,
            "predicted_sampled_latents": hat_z_sampled,
        }

    def forward(
        self,
        x,
        use_consistency_loss=False,
        extended_consistency_loss=False,
    ) -> Dict[str, Any]:
        """
        Compute forward pass of the model.
        Reconstruction: \hat{x} = sum_{i=1}^{n_slots} f(z_i)
        "Imagined" latent vectors: z_tilde = \pi(f^{-1}(\hat{x})})
        "Imagined" images: \hat{x_tilde} = sum_{i=1}^{n_slots} f(z_tilde_i)

        Args:
            x: input image, of shape (batch_size, in_channels, height, width)
            use_consistency_loss: whether to use consistency loss
            extended_consistency_loss: whether to use extended consistency loss

        Returns:
            A tuple containing the following:
                - hat_x: reconstructed input image
                - hat_z: latent vectors for input image
                - figures: figures visualizing each latent vector
                - x_sampled: input image sampled from latent vectors
                - hat_z_sampled: latent vectors for sampled input image
                - figures_sampled: figures visualizing each latent vector for sampled input image
                - z_sampled: sampled latent vectors
                - hat_x_sampled: reconstructed sampled input image
        """
        hat_z = self.encoder(x)
        hat_x, figures = self.decoder(hat_z)

        output_dict = {
            "reconstructed_image": hat_x,
            "predicted_latents": hat_z,
            "reconstructed_figures": figures.permute(1, 0, 2, 3, 4),
        }
        # we always want to look at the consistency loss, but we not always want to backpropagate through consistency part
        consistency_pass_dict = self.consistency_pass(
            hat_z, figures, use_consistency_loss, extended_consistency_loss
        )

        output_dict.update(consistency_pass_dict)
        return output_dict


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
