from contextlib import nullcontext
from dataclasses import dataclass
from typing import List

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch import distributions as dists
from torch import nn

from src.models.ocl.base_model import BaseModel
from .component_vae import ComponentVAE
from .geco import GECO
from .mask_vae import MaskVAE

from src.utils.training_utils import sample_z_from_latents


@dataclass(eq=False, repr=False)
class Genesis(BaseModel):
    sigma_recon: float
    geco_goal_constant: float
    geco_step_size: float
    geco_alpha: float
    geco_init: float
    geco_min: float
    geco_speedup: float
    mask_vae_params: DictConfig
    component_vae_params: DictConfig
    input_channels: int = 3
    model_name: str = "genesis"

    def __post_init__(self):
        super().__post_init__()
        self.mask_latent_size = self.mask_vae_params.latent_size
        self.hidden_state_lstm = self.mask_vae_params.hidden_state_lstm
        self.component_latent_size = self.component_vae_params.latent_size
        self.prior_component = nn.Sequential(
            nn.Linear(self.mask_latent_size, self.hidden_state_lstm),
            nn.ELU(),
            nn.Linear(self.hidden_state_lstm, self.hidden_state_lstm),
            nn.ELU(),
            nn.Linear(self.hidden_state_lstm, self.component_latent_size * 2),
        )
        self.prior_autoregressive_mask = nn.LSTM(
            self.mask_latent_size, self.hidden_state_lstm
        )
        self.prior_linear_mask = nn.Linear(
            self.hidden_state_lstm, self.mask_latent_size * 2
        )
        self.mask_vae_params.update(num_slots=self.num_slots)
        self.mask_vae = MaskVAE(**self.mask_vae_params)
        self.component_vae = ComponentVAE(**self.component_vae_params)

        self.geco_goal_constant *= 3 * self.width * self.height
        self.geco_step_size *= 64**2 / (self.width * self.height)
        self.geco_speedup = self.geco_speedup

        self.geco = GECO(
            self.geco_goal_constant,
            self.geco_step_size,
            self.geco_alpha,
            self.geco_init,
            self.geco_min,
            self.geco_speedup,
        )

    @property
    def slot_size(self) -> int:
        return self.component_latent_size + self.mask_latent_size

    @staticmethod
    def sigma_parameterization(s: Tensor) -> Tensor:
        return (s + 4.0).sigmoid() + 1e-4

    def compute_mask_kl(self, qz_mask: Tensor, z_mask: Tensor) -> Tensor:
        bs = z_mask.shape[0]

        # Autoregressive mask prior for each slot
        pz_mask = [dists.Normal(0, 1)]

        # Permute and remove last slot: (B, nslots, dim) -> (nslots-1, B, z_mask_dim)
        rnn_input = z_mask.permute(1, 0, 2)[:-1]

        # RNN state shape: (nslots-1, B, hidden_dim)
        rnn_state, _ = self.prior_autoregressive_mask(rnn_input)

        # Mu and sigma shape: (nslots-1, B, z_mask_dim)
        pz_mask_mu, pz_mask_sigma = self.prior_linear_mask(rnn_state).chunk(2, dim=-1)
        pz_mask_mu = pz_mask_mu.tanh()
        pz_mask_sigma = self.sigma_parameterization(pz_mask_sigma)

        # First prior mask is N(0, 1). Here we append priors for all other slots.
        for i in range(self.num_slots - 1):
            pz_mask.append(dists.Normal(pz_mask_mu[i], pz_mask_sigma[i]))
        # assert len(pz_mask) == len(qz_mask) == z_mask.size(1) == self.num_slots

        # Compute KL for each slot and return the sum.
        mask_kl = torch.zeros(bs, device=z_mask.device)
        for i in range(self.num_slots):
            mask_kl = mask_kl + (
                qz_mask[i].log_prob(z_mask[:, i]) - pz_mask[i].log_prob(z_mask[:, i])
            ).sum(dim=1)
        return mask_kl.mean(dim=0)

    def compute_component_kl(
        self,
        qz_component: List[dists.Distribution],
        z_mask: Tensor,
        z_component: Tensor,
    ) -> Tensor:
        bs = z_mask.shape[0]

        # Mean and std of component prior: shape (B, nslots, z_mask_dim)
        mu, sigma = self.prior_component(z_mask).chunk(2, dim=-1)
        mu = mu.tanh()
        sigma = self.sigma_parameterization(sigma)

        assert mu.shape == (
            z_component.shape[0],
            self.num_slots,
            self.component_latent_size,
        )
        pz_component = [dists.Normal(mu[:, i], sigma[:, i]) for i in range(mu.size(1))]
        comp_kl = torch.zeros(bs, device=z_mask.device)
        for i in range(len(pz_component)):
            comp_kl = (
                comp_kl
                + qz_component[i].log_prob(z_component[:, i]).sum(dim=1)
                - pz_component[i].log_prob(z_component[:, i]).sum(dim=1)
            )
        return comp_kl.mean(dim=0)

    def compute_recon_loss(
        self, x: Tensor, x_recon_comp: Tensor, log_masks: Tensor
    ) -> Tensor:
        assert (
            x_recon_comp.shape
            == (len(x), self.num_slots, 3, self.width, self.height)
            == x.shape
        )
        assert log_masks.shape == (
            len(x),
            self.num_slots,
            1,
            self.width,
            self.height,
        )
        recon_dist = dists.Normal(x_recon_comp, self.sigma_recon)
        log_p = recon_dist.log_prob(x)
        log_mx = log_p + log_masks
        log_mx = -log_mx.logsumexp(dim=1)  # over slots
        return log_mx.mean(dim=0).sum()

    @staticmethod
    def _flatten_slots(x: Tensor) -> Tensor:
        return x.flatten(0, 1)

    def _unflatten_slots(self, x: Tensor) -> Tensor:
        return x.view(-1, self.num_slots, *x.shape[1:])

    def component_vae_fwd(self, x: Tensor) -> dict:
        # (B, num slots, 4, H, W)
        assert x.dim() == 5
        assert x.shape[1] == self.num_slots
        assert x.shape[2] == self.input_channels + 1

        x = self._flatten_slots(x)
        vae_out = self.component_vae(x)
        for name in ["mu", "sigma", "z", "recon"]:
            vae_out[name] = self._unflatten_slots(vae_out[name])

        # Outputs have shape (B, num slots, ...)
        return vae_out

    def compute_loss(self, x, component_vae_out, mask_vae_out):
        # Compute KL
        qz_component = [
            dists.Normal(
                component_vae_out["mu"][:, i], component_vae_out["sigma"][:, i]
            )
            for i in range(self.num_slots)
        ]
        masks_kl = self.compute_mask_kl(mask_vae_out["qz"], mask_vae_out["z"])
        component_kl = self.compute_component_kl(
            qz_component, mask_vae_out["z"], component_vae_out["z"]
        )
        kl_loss = masks_kl + component_kl

        recon_loss = self.compute_recon_loss(
            x, component_vae_out["recon"], mask_vae_out["log_masks_hat"]
        )
        loss_value = self.geco.loss(recon_loss, kl_loss)

        return loss_value, recon_loss, kl_loss

    def encode(self, x: Tensor):
        # Forward through mask VAE
        mask_vae_out = self.mask_vae(x)  # (B, num slots, 1, H, W)

        # Expand slots: shape (B, num slots, 3, H, W)
        x = x.unsqueeze(1).repeat(1, self.num_slots, 1, 1, 1)
        x = torch.cat([x, mask_vae_out["log_masks_hat"]], dim=2)
        x = self._flatten_slots(x)
        z, mu, sigma = self.component_vae.encode(x)

        component_vae_out = {
            "z": self._unflatten_slots(z),
            "mu": self._unflatten_slots(mu),
            "sigma": self._unflatten_slots(sigma),
        }
        return component_vae_out, mask_vae_out

    def decode(self, z: Tensor):
        z = self._flatten_slots(z)
        xhs = self.component_vae.decode(z)
        xhs = self._unflatten_slots(xhs)
        return xhs

    def forward(
        self,
        x: Tensor,
        use_consistency_loss=False,
        extended_consistency_loss=False,
        detached_latents=False,
    ):
        component_vae_out, mask_vae_out = self.encode(x)
        xhs = self.decode(component_vae_out["z"])
        component_vae_out["recon"] = xhs
        loss_value, recon_loss, kl_loss = self.compute_loss(
            x.unsqueeze(1).repeat(1, self.num_slots, 1, 1, 1),
            component_vae_out,
            mask_vae_out,
        )
        figures = xhs * mask_vae_out["log_masks_hat"].exp()
        images = figures.sum(dim=1)

        mu = torch.cat([mask_vae_out["mu"], component_vae_out["mu"]], dim=2)
        # mu = component_vae_out["mu"]
        with nullcontext() if use_consistency_loss else torch.no_grad():
            z_sampled = sample_z_from_latents(mu.detach())
            z_sampled_component = z_sampled[:, :, self.mask_latent_size :]
            # z_sampled_component = z_sampled
            with torch.no_grad() if detached_latents else nullcontext():
                xhs_sampled = self.decode(z_sampled_component)
                figures_sampled = xhs_sampled * mask_vae_out["log_masks_hat"].exp()
                x_sampled = figures_sampled.sum(dim=1)

            component_vae_out_sampled, mask_vae_out_sampled = self.encode(x_sampled)
            hat_z_sampled = torch.cat(
                [mask_vae_out_sampled["mu"], component_vae_out_sampled["mu"]], dim=2
            )
            # hat_z_sampled = component_vae_out_sampled["mu"]
            with nullcontext() if extended_consistency_loss else torch.no_grad():
                hat_xhs_sampled = self.decode(component_vae_out_sampled["z"])
                hat_x_sampled = (
                    hat_xhs_sampled * mask_vae_out_sampled["log_masks_hat"].exp()
                )
                hat_x_sampled = hat_x_sampled.sum(dim=1)

        figures = [
            figures.squeeze()[:, slot_i, ...] for slot_i in range(self.num_slots)
        ]
        figures_sampled = [
            figures_sampled.squeeze()[:, slot_i, ...]
            for slot_i in range(self.num_slots)
        ]
        return (
            images,
            mu,
            figures,
            x_sampled,
            hat_z_sampled,
            figures_sampled,
            z_sampled,
            hat_x_sampled,
            loss_value,
        )
