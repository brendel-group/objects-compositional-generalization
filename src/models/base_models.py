from contextlib import nullcontext
from typing import Any, Dict

import torch
from src.utils.training_utils import sample_z_from_latents

from . import utils


class SlotEncoder(torch.nn.Module):
    def __init__(self, in_channels, n_slots, n_slot_latents, dataset_name):
        super(SlotEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = utils.get_encoder(
            in_channels, n_slots * n_slot_latents, dataset_name
        )

    def forward(self, x):
        x = self.encoder(x)
        out = x.view(-1, self.n_slots, self.n_slot_latents)
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
        dataset_name: str,
    ) -> None:
        super(SlotMLPAdditiveDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.decoder = utils.get_decoder(n_slot_latents, in_channels, dataset_name)
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
        dataset_name: str,
    ) -> None:
        super(SlotMLPMonolithicDecoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.decoder = utils.get_decoder(
            n_slots * n_slot_latents, in_channels, dataset_name
        )
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
        dataset_name: str,
    ) -> None:
        super(SlotMLPMonolithic, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents, dataset_name)
        self.decoder = SlotMLPMonolithicDecoder(
            in_channels, n_slots, n_slot_latents, dataset_name
        )
        self.model_name = "SlotMLPMonolithic"

    def forward(self, x, **kwargs):
        latents = self.encoder(x)
        latents = latents.view(-1, self.n_slots * self.n_slot_latents)
        image = self.decoder(latents)
        latents = latents.view(-1, self.n_slots, self.n_slot_latents)
        return {
            "reconstructed_image": image,
            "predicted_latents": latents,
        }


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
        no_overlap: bool = False,
        dataset_name: str = "dsprites",
    ) -> None:
        super(SlotMLPAdditive, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.no_overlap = no_overlap
        self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents, dataset_name)
        self.decoder = SlotMLPAdditiveDecoder(
            in_channels, n_slots, n_slot_latents, dataset_name
        )
        self.model_name = "SlotMLPAdditive"

    def consistency_pass(
        self,
        hat_z,
        use_consistency_loss,
        z_sampled=None,
    ):
        # getting imaginary samples
        with torch.no_grad():
            if z_sampled is None:
                z_sampled, indices = sample_z_from_latents(hat_z.detach())
            x_sampled, figures_sampled = self.decoder(z_sampled)
            x_sampled = torch.clamp(x_sampled, 0, 1)

        # encoder pass
        with nullcontext() if (use_consistency_loss) else torch.no_grad():
            hat_z_sampled = self.encoder(x_sampled)

        # second decoder pass - for debugging purposes
        with torch.no_grad():
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
        true_latents=None,
        true_figures=None,
        not_ignore_consistency=True,
    ) -> Dict[str, Any]:
        """
        Compute forward pass of the model.
        Reconstruction: \hat{x} = sum_{i=1}^{n_slots} f(z_i)
        "Imagined" latent vectors: z_tilde = \pi(f^{-1}(\hat{x})})
        "Imagined" images: \hat{x_tilde} = sum_{i=1}^{n_slots} f(z_tilde_i)

        Args:
            x: input image, of shape (batch_size, in_channels, height, width)
            use_consistency_loss: whether to use consistency loss
            true_latents: true latents for input image
            true_figures: true figures for input image
            not_ignore_consistency: whether to ignore consistency loss

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
        if (
            true_latents is not None
            and true_figures is not None
            and self.no_overlap
            and use_consistency_loss
        ):
            with torch.no_grad():
                z_sampled = sample_z_from_latents_no_overlap(
                    true_latents, hat_z, true_figures, figures, hat_z.device
                )
        else:
            z_sampled = None

        if not_ignore_consistency:
            consistency_pass_dict = self.consistency_pass(
                hat_z,
                figures,
                use_consistency_loss,
                z_sampled=z_sampled,
            )
        else:
            consistency_pass_dict = {}

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
        dataset_name: str,
    ) -> None:
        super(SlotMLPEncoder, self).__init__()
        self.n_slots = n_slots
        self.n_slot_latents = n_slot_latents
        self.encoder = SlotEncoder(in_channels, n_slots, n_slot_latents, dataset_name)
        self.model_name = "SlotMLPEncoder"

    def forward(self, x):
        latents = self.encoder(x)
        return latents
