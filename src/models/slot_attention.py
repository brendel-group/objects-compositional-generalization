import torch
from torch import nn
from torch.nn import init

import models_utils


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super(SlotAttention, self).__init__()

        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device=device)

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
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SoftPositionEmbedding(nn.Module):
    def __init__(self, hidden_size, resolution):
        super(SoftPositionEmbedding, self).__init__()
        self.grid = models_utils.build_grid(resolution)
        self.linear = nn.Linear(2, hidden_size)

    def forward(self, x):
        return x + self.linear(self.grid)


class SlotAttentionModel(nn.Module):
    def __init__(self, num_slots, resolution, num_iterations, in_channels=3):
        super(SlotAttentionModel, self).__init__()
        self.num_slots = num_slots
        self.resolution = resolution
        self.num_iterations = num_iterations
        self.slot_size = 8

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 2, 2),
            nn.ReLU(),
        )

        self.encoder_pos = SoftPositionEmbedding(64, (64, 64))

        self.layer_norm = nn.LayerNorm(64)

        self.slot_attention = SlotAttention(
            num_slots, self.slot_size, num_iterations, hidden_dim=128
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.decoder = models_utils.get_decoder(self.num_slots * self.slot_size, 3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_pos(x)
        x = models_utils.View((-1, 64 * 4 * 4))(x)
        x = self.layer_norm(x)
        x = self.slot_attention(x, self.num_slots)
        x = self.mlp(x)


