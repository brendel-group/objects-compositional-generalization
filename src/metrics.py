"""
Metrics for evaluating models.
"""
from typing import List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from src.model_evaluation import evaluate_model
from torchmetrics import R2Score


def r2_score(
    true_latents: torch.Tensor, predicted_latents: torch.Tensor, indices: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """
    Calculates R2 score. Slots are flattened before calculating R2 score.

    Args:
        true_latents: tensor of shape (batch_size, n_slots, n_latents)
        predicted_latents: tensor of shape (batch_size, n_slots, n_latents)
        indices: tensor of shape (batch_size, n_slots, 2) with indices of matched slots

    Returns:
        avg_r2_score: average R2 score over all latents
        r2_score_raw: R2 score for each latent
    """
    indices = torch.LongTensor(indices)
    predicted_latents = predicted_latents.detach().cpu()
    true_latents = true_latents.detach().cpu()

    # shuffling predicted latents to match true latents
    predicted_latents = predicted_latents.gather(
        1,
        indices[:, :, 1].unsqueeze(-1).expand(-1, -1, predicted_latents.shape[-1]),
    )
    true_latents = true_latents.flatten(start_dim=1)
    predicted_latents = predicted_latents.flatten(start_dim=1)
    r2 = R2Score(true_latents.shape[1], multioutput="raw_values")
    r2_score_raw = r2(predicted_latents, true_latents)
    r2_score_raw[torch.isinf(r2_score_raw)] = torch.nan
    avg_r2_score = torch.nanmean(r2_score_raw).item()
    return avg_r2_score, r2_score_raw


def image_r2_score(true_images: torch.Tensor, predicted_images: torch.Tensor) -> float:
    """
    Calculates R2 score for images. Used for image reconstruction evaluation.

    Args:
        true_images: tensor of shape (batch_size, n_channels, height, width)
        predicted_images: tensor of shape (batch_size, n_channels, height, width)

    Returns:
        reconstruction_error: R2 score
    """

    r2_vw = R2Score(
        num_outputs=np.prod(true_images.shape[1:]), multioutput="variance_weighted"
    ).to(true_images.device)

    # add eps to avoid division by zero
    true_images += 1e-7

    reconstruction_error = r2_vw(
        predicted_images.reshape(predicted_images.shape[0], -1),
        true_images.reshape(true_images.shape[0], -1),
    )

    return reconstruction_error


def hungarian_slots_loss(
    true_latents: torch.Tensor,
    predicted_latents: torch.Tensor,
    device: str = "cpu",
    p: int = 2,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # Normalizing the latents
    true_latents = (true_latents - true_latents.mean()) / (true_latents.std() + 1e-8)
    predicted_latents = (predicted_latents - predicted_latents.mean()) / (
        predicted_latents.std() + 1e-8
    )

    pairwise_cost = (
        torch.cdist(true_latents, predicted_latents, p=p).transpose(-1, -2).pow(p)
    )

    indices = np.array(
        list(map(linear_sum_assignment, pairwise_cost.detach().cpu().numpy()))
    )  # applying hungarian algorithm to every sample in batch
    transposed_indices = torch.from_numpy(
        np.transpose(indices, axes=(0, 2, 1))
    )  # these indexes are showing how I need to shuffle the g.t. latents to match the predicted latents

    # extracting the cost of the matched slots; this code is a bit ugly, idk what is the nice way to do it
    loss = torch.gather(pairwise_cost, 2, transposed_indices.to(device))[:, :, -1].sum(
        1
    )  # sum along slots

    # taking the inverse, to match the predicted latents to the g.t.
    inverse_indices = torch.argsort(transposed_indices, dim=1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError("Reduction type not supported.")

    return loss, inverse_indices


def reconstruction_loss(
    target: torch.Tensor, prediction: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes the reconstruction loss.

    Args:
        target: tensor of shape [batch_size, *]
        prediction: tensor of shape [batch_size, *]
        reduction: "mean" or "sum"
    """
    loss = (target - prediction).square()

    if reduction == "mean":
        loss = loss.mean(dim=0)
    elif reduction == "sum":
        loss = loss.sum(dim=0)

    return loss.sum()


def ari(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    num_ignored_objects: int = 1,
) -> torch.Tensor:
    """Computes the ARI score.

    Args:
        true_mask: tensor of shape [batch_size, n_objects, *] where values go from 0 to the number of objects.
        pred_mask:  tensor of shape [batch_size, n_objects, *] where values go from 0 to the number of objects.
        num_ignored_objects: number of objects (in ground-truth mask) to be ignored when computing ARI.
        (Usually 1 for background.)

    Returns:
        a vector of ARI scores, of shape [batch_size, ].
    """
    true_mask = true_mask.cpu().argmax(dim=1, keepdim=True).squeeze(2)
    pred_mask = pred_mask.cpu().argmax(dim=1, keepdim=True).squeeze(2)

    true_mask = true_mask.flatten(1)
    pred_mask = pred_mask.flatten(1)

    not_bg = true_mask >= num_ignored_objects

    result = []
    batch_size = len(true_mask)
    for i in range(batch_size):
        ari_value = adjusted_rand_score(
            true_mask[i][not_bg[i]], pred_mask[i][not_bg[i]]
        )
        result.append(ari_value)
    return torch.tensor(result).mean()


def identifiability_score(
    model: torch.nn.Module,
    test_id_loader: torch.utils.data.DataLoader,
    test_ood_loader: torch.utils.data.DataLoader,
    categorical_dimensions: List[int],
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Computes identifiability score.
    First fits a model on the ID latents and then evaluates it on the OOD latents.
    For more details see: https://arxiv.org/abs/2305.14229

    Brady, Jack, et al. "Provably Learning Object-Centric Representations."
    arXiv preprint arXiv:2305.14229 (2023).

    Args:
        model: model to evaluate
        test_id_loader: dataloader for ID test set
        test_ood_loader: dataloader for OOD test set
        categorical_dimensions: list of categorical dimensions
        device: device to run on

    Returns:
        id_score_id: identifiability score for ID test set
        id_score_ood: identifiability score for OOD test set
    """

    # collect test set latents and predicted latents
    z_true_id = []
    z_pred_id = []
    z_true_ood = []
    z_pred_ood = []
    with torch.no_grad():
        for images, true_latents in test_id_loader:
            images = images[:, -1, ...].to(device)
            true_latents = true_latents.to(device)
            z_true_id.append(true_latents)
            output = model(images, not_ignore_consistency=False)
            z_pred_id.append(output["predicted_latents"])

    z_true_id = torch.cat(z_true_id, dim=0)
    z_pred_id = torch.cat(z_pred_id, dim=0)

    with torch.no_grad():
        for images, true_latents in test_ood_loader:
            images = images[:, -1, ...].to(device)
            true_latents = true_latents.to(device)
            z_true_ood.append(true_latents)
            output = model(images, not_ignore_consistency=False)
            z_pred_ood.append(output["predicted_latents"])

    z_true_ood = torch.cat(z_true_ood, dim=0)
    z_pred_ood = torch.cat(z_pred_ood, dim=0)

    # calculate identifiability score
    identifiability_score_id = evaluate_model(
        z_true_id,
        z_pred_id,
        categorical_dimensions,
        max_training_epochs=100,
        model_depth=5,
        train_val_test_split=(0.7, 0.1, 0.2),
        verbose=2,
        standard_scale=True,
        z_mask_values=0,
    )
    (
        performance_id,
        continuous_performance_id,
        categorical_performance_id,
        (r2_scores_id, accuracies_id),
        (ceiling_r2_scores_id, ceiling_accuracies_id),
        models_id,
    ) = identifiability_score_id

    identifiability_score_ood = evaluate_model(
        z_true_ood,
        z_pred_ood,
        categorical_dimensions,
        max_training_epochs=100,
        model_depth=5,
        train_val_test_split=(0.7, 0.1, 0.2),
        verbose=2,
        standard_scale=True,
        z_mask_values=0,
        provided_models=models_id,
    )

    (
        performance_ood,
        continuous_performance_ood,
        categorical_performance_ood,
        (r2_scores_ood, accuracies_ood),
        (ceiling_r2_scores_ood, ceiling_accuracies_ood),
        _,
    ) = identifiability_score_ood

    id_score_id = r2_scores_id[0] * (
        z_true_id.shape[-1] - len(categorical_dimensions)
    ) + accuracies_id[0] * len(categorical_dimensions)
    id_score_id /= z_true_id.shape[-1]

    id_score_ood = r2_scores_ood[0] * (
        z_true_ood.shape[-1] - len(categorical_dimensions)
    ) + accuracies_ood[0] * len(categorical_dimensions)
    id_score_ood /= z_true_ood.shape[-1]

    return id_score_id, id_score_ood
