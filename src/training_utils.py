import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import R2Score


def calculate_r2_score(true_latents, predicted_latents, inds):
    """Calculate R2 score. Slots are flattened before calculating R2 score."""
    inds = torch.LongTensor(inds)
    predicted_latents = predicted_latents.detach().cpu()
    true_latents = true_latents.detach().cpu()

    for i in range(true_latents.shape[0]):
        # shuffling predicted latents to match true latents
        predicted_latents[i, :] = predicted_latents[i, inds[i, :, 1], ...]

    true_latents = true_latents.view(true_latents.shape[0], -1)
    predicted_latents = predicted_latents.reshape(predicted_latents.shape[0], -1)

    r2 = R2Score(true_latents.shape[1])
    return r2(predicted_latents, true_latents)


def matched_slots_loss(
    true_latents, predicted_latents, device="cpu", p=2, reduction="sum"
):
    """
    Computes pairwise distance between slots, matches slots with Hungarian algorithm and outputs
    sum of distances ^ p.
    """
    pairwise_cost = torch.pow(
        torch.cdist(true_latents, predicted_latents, p=p).transpose(-1, -2), p
    )
    indices = np.array(
        list(map(linear_sum_assignment, pairwise_cost.detach().cpu().numpy()))
    )  # applying hungarian algorithm to every sample in batch
    transposed_indices = torch.from_numpy(np.transpose(indices, axes=(0, 2, 1)))

    # extracting the cost of the matched slots; this code is a bit ugly, idk what is the nice way to do it
    output = torch.gather(pairwise_cost, 2, transposed_indices.to(device))[
        :, :, -1
    ].sum(1)

    if reduction == "mean":
        output = output.mean()
    elif reduction == "sum":
        output = output.sum()
    else:
        raise ValueError("Reduction type not supported.")

    return output, transposed_indices


def collate_fn_normalizer(batch, bias=0, scale=1):
    """Normalize latents target to [0, 1]."""
    images, latents = zip(*batch)
    latents = torch.stack(latents)
    latents = (latents - bias) / scale
    return torch.stack(images), latents
