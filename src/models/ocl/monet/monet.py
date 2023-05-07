from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

import torch
import torch.distributions as dists
from omegaconf import DictConfig
from torch import Tensor, nn

from src.models.ocl.base_model import BaseModel
from src.models.ocl.shared.encoder_decoder import BroadcastDecoderNet, EncoderNet
from src.models.ocl.shared.unet import UNet
from src.utils.training_utils import sample_z_from_latents


@dataclass(eq=False, repr=False)
class MONet(BaseModel):
    latent_size: int

    num_blocks_unet: int
    beta_kl: float
    gamma: float

    encoder_params: DictConfig
    decoder_params: DictConfig

    model_name: str = "monet"
    input_channels: int = 3
    bg_sigma: float = 0.09
    fg_sigma: float = 0.11

    prior_mean: float = 0.0
    prior_std: float = 1.0

    encoder: EncoderNet = field(init=False)
    decoder: BroadcastDecoderNet = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.attention = AttentionNet(self.input_channels, self.num_blocks_unet)
        self.encoder_params.update(width=self.width, height=self.height)
        self.encoder = EncoderNet(**self.encoder_params)
        self.decoder = BroadcastDecoderNet(**self.decoder_params)
        self.prior_dist = dists.Normal(self.prior_mean, self.prior_std)

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def forward(
        self,
        x: Tensor,
        use_consistency_loss=False,
        extended_consistency_loss=False,
        detached_latents=False,
    ) -> Dict[str, Any]:
        # input: (B, 3, H, W)

        zs, kl_zs, slot_means, masks, log_masks = self.encoding(x)
        recons, xhs, masks_pred, slots = self.decoding(zs)
        loss = self._compute_loss(x, slots, masks, log_masks, masks_pred, kl_zs)

        with nullcontext() if use_consistency_loss else torch.no_grad():
            z_sampled = sample_z_from_latents(slot_means.detach())
            with torch.no_grad() if detached_latents else nullcontext():
                x_sampled, figures_sampled, masks_sampled, _ = self.decoding(z_sampled)

            zs, _, hat_z_sampled, _, _ = self.encoding(x_sampled)

            with nullcontext() if extended_consistency_loss else torch.no_grad():
                hat_x_sampled, _, _, _ = self.decoding(zs)

        figures = [xhs.squeeze()[:, slot_i, ...] for slot_i in range(self.num_slots)]
        figures_sampled = [
            figures_sampled.squeeze()[:, slot_i, ...]
            for slot_i in range(self.num_slots)
        ]
        return {
            "reconstructed_image": recons,
            "predicted_latents": slot_means,
            "reconstructed_figures": figures,
            "reconstructed_masks": masks_pred,
            "sampled_image": x_sampled,
            "sampled_figures": figures_sampled,
            "sampled_latents": z_sampled,
            "reconstructed_sampled_image": hat_x_sampled,
            "predicted_sampled_latents": hat_z_sampled,
            "loss": loss,
        }
        # return (
        #     recons,
        #     slot_means,
        #     figures,
        #     x_sampled,
        #     hat_z_sampled,
        #     figures_sampled,
        #     z_sampled,
        #     hat_x_sampled,
        #     loss,
        # )

    def encoding(self, x):
        # Forward pass through recurrent attention network.
        log_masks = self._attention_process(x)  # (B, num slots, H, W)
        masks = log_masks.exp()

        # Component VAE forward pass.
        zs, kl_zs, slot_means = self._encode(x, log_masks)
        return zs, kl_zs, slot_means, masks, log_masks

    def decoding(self, zs: Tensor):
        slots, masks_pred = self._decode(zs)
        xhs = slots * masks_pred.unsqueeze(2)
        recons = xhs.sum(dim=1)
        return recons, xhs, masks_pred, slots

    def _compute_loss(self, x, slots, masks, log_masks, masks_pred, kl_zs):
        # Overall (scalar) negative log likelihood: -log p(x|z).
        neg_log_pxs = self._compute_likelihood(x, slots, log_masks)
        # KL for reconstructed masks.
        mask_kl = self._compute_mask_kl(masks, masks_pred)
        loss = neg_log_pxs + self.beta_kl * kl_zs + self.gamma * mask_kl
        return loss

    def _attention_process(self, x: Tensor) -> Tensor:
        scope_shape = list(x.shape)
        scope_shape[1] = 1
        log_scope = torch.zeros(scope_shape, device=x.device)
        log_masks = []
        for i in range(self.num_slots - 1):
            # Mask and scope: (B, 1, H, W)
            log_mask, log_scope = self.attention(x, log_scope)
            log_masks.append(log_mask)
        log_masks.append(log_scope)
        log_masks = torch.cat(log_masks, dim=1)  # (B, num slots, H, W)
        return log_masks

    def _compute_likelihood(
        self, x: Tensor, slots: Tensor, log_masks: Tensor
    ) -> Tensor:
        sigma = torch.full(
            [
                self.num_slots,
            ],
            self.fg_sigma,
            device=x.device,
        )
        sigma[0] = self.bg_sigma
        sigma = sigma.reshape([1, self.num_slots, 1, 1, 1])
        dist = dists.Normal(slots, sigma)
        log_pxs_masked = log_masks.unsqueeze(2) + dist.log_prob(x.unsqueeze(1))

        # Global negative log likelihood p(x|z): scalar.
        neg_log_pxs = -log_pxs_masked.logsumexp(dim=1).mean(dim=0).sum()

        return neg_log_pxs

    def _compute_mask_kl(self, masks: Tensor, masks_pred: Tensor) -> Tensor:
        bs = len(masks)
        d_masks = self._make_mask_distribution(masks)
        d_masks_pred = self._make_mask_distribution(masks_pred)
        kl_masks = dists.kl_divergence(d_masks, d_masks_pred)
        kl_masks = kl_masks.sum() / bs
        return kl_masks

    @staticmethod
    def _make_mask_distribution(masks: Tensor) -> dists.Distribution:
        flat_masks = masks.permute(0, 2, 3, 1).flatten(0, 2)  # (B*H*W, n_slots)
        flat_masks = flat_masks.clamp_min(1e-5)
        d_masks = dists.Categorical(probs=flat_masks)
        return d_masks

    def _encode(self, x: Tensor, log_masks: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (B, 3, H, W) -> (B, num slots, 3, H, W)
        # log_masks: (B, num slots, H, W) -> (B, num slots, 1, H, W)
        x = x.unsqueeze(1).repeat(1, self.num_slots, 1, 1, 1)
        log_masks = log_masks.unsqueeze(2)

        # Encoder input: (B * num slots, RGB+mask, H, W).
        encoder_input = torch.cat([x, log_masks], 2).flatten(0, 1)

        # Encode and reshape parameters to (B, num slots, latent dim).
        mean, log_sigma = self.encoder(encoder_input).chunk(2, dim=1)
        sigma = log_sigma.exp()
        mean = mean.unflatten(0, [x.shape[0], self.num_slots])  # (B, num_slots, D)
        sigma = sigma.unflatten(0, [x.shape[0], self.num_slots])  # (B, num_slots, D)

        # Return mean, sample, and KL.
        latent_normal = dists.Normal(mean, sigma)
        kl_z = dists.kl_divergence(latent_normal, self.prior_dist)
        kl_z = kl_z.sum(dim=[1, 2]).mean(0)  # sum over latent dimensions and slots
        z = latent_normal.rsample()  # (B, num_slots, D)
        return z, kl_z, mean

    def _decode(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = zs.shape[0]
        zs = zs.flatten(0, 1)  # (B * num slots, D)
        decoder_output = self.decoder(zs)

        # (B * num slots, 3, H, W)
        slots_recon = decoder_output[:, :3].sigmoid()
        # (B, num slots, 3, H, W)
        slots_recon = slots_recon.unflatten(0, [batch_size, self.num_slots])

        # (B * num slots, 1, H, W)
        mask_pred = decoder_output[:, 3:]
        # (B, num slots, H, W)
        mask_pred = mask_pred.unflatten(0, [batch_size, self.num_slots]).squeeze(2)
        mask_pred = mask_pred.softmax(dim=1)

        return slots_recon, mask_pred


class AttentionNet(nn.Module):
    def __init__(self, input_channels: int, num_blocks: int):
        super().__init__()
        self.unet = UNet(input_channels, num_blocks)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor, log_scope: Tensor) -> Tuple[Tensor, Tensor]:
        inp = torch.cat((x, log_scope), 1)
        logits = self.unet(inp)
        # logits = self.last_conv(logits)
        log_alpha = self.log_softmax(logits)
        log_mask = log_scope + log_alpha[:, 0:1]  # log (scope * alpha)
        log_scope = log_scope + log_alpha[:, 1:2]  # log (scope * (1-alpha))
        return log_mask, log_scope
