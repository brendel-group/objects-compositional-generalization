from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import R2Score


def calculate_r2_score(
    true_latents: torch.Tensor, predicted_latents: torch.Tensor, inds: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Calculate R2 score. Slots are flattened before calculating R2 score."""
    inds = torch.LongTensor(inds)
    predicted_latents = predicted_latents.detach().cpu()
    true_latents = true_latents.detach().cpu()

    for i in range(true_latents.shape[0]):
        # shuffling predicted latents to match true latents
        predicted_latents[i, :] = predicted_latents[i, inds[i, :, 1], ...]

    true_latents = true_latents.view(true_latents.shape[0], -1)
    predicted_latents = predicted_latents.reshape(predicted_latents.shape[0], -1)

    r2 = R2Score(true_latents.shape[1], multioutput="raw_values")
    r2_score_raw = r2(predicted_latents, true_latents)
    return r2_score_raw.mean().item(), r2_score_raw


def matched_slots_loss(
    true_latents: torch.Tensor,
    predicted_latents: torch.Tensor,
    device: str = "cpu",
    p: int = 2,
    reduction: str = "sum",
):
    """
    Computes pairwise distance between slots, matches slots with Hungarian algorithm and outputs
    sum of distances ^ p.

    Args:
        true_latents: tensor of shape (batch_size, n_slots, n_latents)
        predicted_latents: tensor of shape (batch_size, n_slots, n_latents)
        device: device to run on
        p: for l-p distance, i.e. ||x - y||_p^p
        reduction: "sum" for distance sum over all slots or "mean" for average distance

    Returns:
        loss: sum/mean of distances ^ p
        transposed_indices: indices of matched slots (later used for R2 score calculation)
    """
    pairwise_cost = torch.pow(
        torch.cdist(true_latents, predicted_latents, p=p).transpose(-1, -2), p
    )
    indices = np.array(
        list(map(linear_sum_assignment, pairwise_cost.detach().cpu().numpy()))
    )  # applying hungarian algorithm to every sample in batch
    transposed_indices = torch.from_numpy(np.transpose(indices, axes=(0, 2, 1)))

    # extracting the cost of the matched slots; this code is a bit ugly, idk what is the nice way to do it
    loss = torch.gather(pairwise_cost, 2, transposed_indices.to(device))[:, :, -1].sum(
        1
    )

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError("Reduction type not supported.")

    return loss, transposed_indices


def collate_fn_normalizer(batch, bias=0, scale=1):
    """Normalize latents target to [0, 1]. Used in dataloader."""
    images, latents = zip(*batch)
    latents = torch.stack(latents)
    latents = (latents - bias) / scale
    return torch.stack(images), latents
