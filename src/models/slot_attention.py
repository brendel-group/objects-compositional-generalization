"""
Slot Attention-based autoencoder; implementation is adapted from 
https://github.com/addtt/object-centric-library.
"""

from contextlib import nullcontext
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.training_utils import sample_z_from_latents


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        num_slots,
        num_iterations,
        hid_dim,
        softmax=True,
        sampling=True,
    ):
        """
        Builds the Slot Attention-based auto-encoder.

        Args:
            num_slots: Number of slots in Slot Attention.
            num_iterations: Number of iterations in Slot Attention.
            hid_dim: Hidden dimension of Slot Attention.
        """
        super().__init__()
        self.model_name = "SlotAttention"
        self.hid_dim = hid_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = encoder
        self.decoder_cnn = decoder

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters=self.num_iterations,
            eps=1e-8,
            hidden_dim=64,
            use_sampling=sampling,
        )

        self.spatial_broadcast = 64

        self.use_softmax = softmax

    def encode(self, x):
        # `x` is an image which has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(x)  # CNN Backbone.
        x = F.layer_norm(x, x.shape[1:])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        return self.slot_attention(x)

    def decode(self, hat_z):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        slots = (
            hat_z.reshape((-1, hat_z.shape[-1]))
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat((1, self.spatial_broadcast, self.spatial_broadcast, 1))
        )

        # `out` has shape: [batch_size*num_slots, width, height, num_channels+1].
        out = self.decoder_cnn(slots)

        # Undo combination of slot and batch dimension; split alpha masks.
        reconstructions, masks = out.reshape(
            -1, self.num_slots, out.shape[1], out.shape[2], out.shape[3]
        ).split([3, 1], dim=-1)

        # Normalize alpha masks over slots.
        if self.use_softmax:
            masks = F.softmax(masks, dim=1)
        else:
            masks = torch.sigmoid(masks)
        xhs = reconstructions * masks

        # `hat_x` has shape: [batch_size, num_channels, width, height].
        hat_x = torch.sum(xhs, dim=1).permute(0, 3, 1, 2)

        return hat_x, xhs, masks, reconstructions

    def forward(
        self,
        x,
        use_consistency_loss=False,
        extended_consistency_loss=False,
    ) -> Dict[str, Any]:
        hat_z = self.encode(x)
        hat_x, figures, masks, reconstructions = self.decode(hat_z)

        output_dict = {
            "reconstructed_image": hat_x,
            "predicted_latents": hat_z,
            "reconstructed_figures": figures.permute(1, 0, 4, 2, 3),
            "reconstructed_masks": masks.permute(1, 0, 4, 2, 3),
            "raw_figures": reconstructions.permute(1, 0, 4, 2, 3),
        }

        # we always want to look at the consistency loss, but we not always want to backpropagate through consistency part
        consistency_pass_dict = self.consistency_pass(
            hat_z,
            use_consistency_loss,
        )

        output_dict.update(consistency_pass_dict)
        return output_dict

    def consistency_pass(
        self,
        hat_z,
        use_consistency_loss,
    ):
        # getting imaginary samples
        with torch.no_grad():
            z_sampled, indices = sample_z_from_latents(hat_z.detach())

            (
                x_sampled,
                figures_sampled,
                sampled_masks,
                raw_sampled_figures,
            ) = self.decode(z_sampled)
            x_sampled = x_sampled.clamp(0, 1)

        # encoder pass
        with nullcontext() if (use_consistency_loss) else torch.no_grad():
            hat_z_sampled = self.encode(x_sampled)

        # second decoder pass - for debugging purposes
        with torch.no_grad():
            hat_x_sampled, _, hat_sampled_masks, hat_raw_sampled_figures = self.decode(
                hat_z_sampled
            )

        return {
            "sampled_image": x_sampled,
            "sampled_figures": figures_sampled.permute(1, 0, 4, 2, 3),
            "sampled_latents": z_sampled,
            "sampled_masks": sampled_masks.permute(1, 0, 4, 2, 3),
            "reconstructed_sampled_image": hat_x_sampled,
            "predicted_sampled_latents": hat_z_sampled,
            "raw_sampled_figures": raw_sampled_figures.permute(1, 0, 4, 2, 3),
        }


class SlotAttention(nn.Module):
    def __init__(
        self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128, use_sampling=True
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        self.use_sampling = use_sampling

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1).exp()

        # remove randomness from slots initialization
        if not self.use_sampling:
            g = torch.Generator()
            g.manual_seed(0)

            slots = mu + sigma * torch.normal(0, 1, size=sigma.shape, generator=g).to(
                sigma.device
            )
        else:
            slots = mu + sigma * torch.randn_like(sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots


def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid.to(inputs.device))
        return inputs + grid


class SlotAttentionEncoder(nn.Module):
    def __init__(self, resolution, hid_dim, ch_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(ch_dim, ch_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(ch_dim, ch_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(ch_dim, hid_dim, 5, padding=2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x


class SlotAttentionDecoder(nn.Module):
    def __init__(self, ch_dim, hid_dim, resolution):
        super().__init__()

        self.decoder_initial_size = (64, 64)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(hid_dim, ch_dim, 5, padding=2), nn.ReLU()]
        )
        for i in range(2):
            self.conv_list.append(nn.Conv2d(ch_dim, ch_dim, 5, padding=2))
            self.conv_list.append(nn.ReLU())

        self.conv_list.append(nn.Conv2d(ch_dim, 4, 3, padding=1))

        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        for conv_layer in self.conv_list:
            x = conv_layer(x)

        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x


class PositionalEmbedding(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        east = torch.linspace(0, 1, width).repeat(height)
        west = torch.linspace(1, 0, width).repeat(height)
        south = torch.linspace(0, 1, height).repeat(width)
        north = torch.linspace(1, 0, height).repeat(width)
        east = east.reshape(height, width)
        west = west.reshape(height, width)
        south = south.reshape(width, height).T
        north = north.reshape(width, height).T
        # (4, h, w)
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, channels, kernel_size=1)
        self.register_buffer("linear_position_embedding", linear_pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(
            x.size(0), 4, x.size(2), x.size(3)
        )
        x = x + self.channels_map(bs_linear_position_embedding)
        return x
