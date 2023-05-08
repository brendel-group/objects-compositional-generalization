from typing import Tuple

import numpy as np
import torch
import tqdm
from scipy.optimize import linear_sum_assignment
from torch import nn
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
        indices[:, :, 1].unsqueeze(-1).expand(-1, -1, true_latents.shape[-1]),
    )
    true_latents = true_latents.flatten(start_dim=1)
    predicted_latents = predicted_latents.flatten(start_dim=1)
    r2 = R2Score(true_latents.shape[1], multioutput="raw_values")
    r2_score_raw = r2(predicted_latents, true_latents)
    r2_score_raw[torch.isinf(r2_score_raw)] = torch.nan
    avg_r2_score = torch.nanmean(r2_score_raw).item()
    return avg_r2_score, r2_score_raw


def hungarian_slots_loss(
    true_latents: torch.Tensor,
    predicted_latents: torch.Tensor,
    device: str = "cpu",
    p: int = 2,
    reduction: str = "mean",
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
    )  # sum along slots

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError("Reduction type not supported.")

    return loss, transposed_indices


def reconstruction_loss(target, prediction, reduction="mean"):
    loss = (target - prediction).square()

    if reduction == "mean":
        loss = loss.mean(dim=0)
    elif reduction == "sum":
        loss = loss.sum(dim=0)

    return loss.sum()


def identifiability_score(
    model: torch.nn.Module,
    latent_size: int,
    train_loader: torch.utils.data.DataLoader,
    test_id_loader: torch.utils.data.DataLoader,
    test_ood_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
):
    """
    Calculates identifiability score for a model. Identifiability score is calculated as R2 score
    between true and predicted latents after mapping true latents to predicted latents space using
    MLP.

    Args:
        model: model to calculate identifiability score for
        latent_size: size of latent space
        train_loader: train loader
        test_id_loader: test loader for in-distribution data
        test_ood_loader: test loader for out-of-distribution data
        device: device to run on

    Returns:
        (id_score, ood_score): identifiability score for in-distribution and out-of-distribution data
    """

    model.eval()

    input_dim = train_loader.dataset[0][1].shape[1]
    n_slots = train_loader.dataset[0][1].shape[0]

    mlp = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, latent_size),
    ).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()
    best_id_scores = [torch.tensor(-torch.inf), torch.tensor(-torch.inf)]
    for epoch in tqdm.tqdm(range(20)):
        mlp.train()
        for i, (images, true_latents) in enumerate(train_loader):
            optimizer.zero_grad()

            figures = images[:, :-1, ...].to(device)
            images = images[:, -1, ...].to(device).squeeze(1)
            true_latents = true_latents.to(device)

            with torch.no_grad():
                output = model(images)
                predicted_latents = output["predicted_latents"]
                predicted_figures = output["reconstructed_figures"]

            mapped_latents = torch.stack(
                [mlp(true_latents[:, i, :]) for i in range(n_slots)], dim=1
            )

            figures_reshaped = figures.view(figures.shape[0], figures.shape[1], -1)
            predicted_figures = torch.stack(predicted_figures, dim=1)

            predicted_figures_reshaped = predicted_figures.reshape(
                predicted_figures.shape[0], predicted_figures.shape[1], -1
            )
            with torch.no_grad():
                # no loss calculated here, just indices for resolving permutations
                _, transposed_indices = hungarian_slots_loss(
                    figures_reshaped,
                    predicted_figures_reshaped,
                    device=device,
                )
            transposed_indices = transposed_indices.to(device)

            predicted_latents = predicted_latents.gather(
                1,
                transposed_indices[:, :, 1]
                .unsqueeze(-1)
                .expand(-1, -1, mapped_latents.shape[-1]),
            )

            loss = criterion(
                mapped_latents.view(-1, latent_size),
                predicted_latents.view(-1, latent_size),
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        mlp.eval()
        id_scores = []
        for loader in [test_id_loader, test_ood_loader]:
            r2_scores = []
            for i, (images, true_latents) in enumerate(loader):
                figures = images[:, :-1, ...].to(device)
                images = images[:, -1, ...].to(device).squeeze(1)
                true_latents = true_latents.to(device)

                with torch.no_grad():
                    output = model(images)
                    predicted_latents = output["predicted_latents"]
                    predicted_figures = output["reconstructed_figures"]

                mapped_latents = torch.stack(
                    [mlp(true_latents[:, i, :]) for i in range(n_slots)], dim=1
                )

                figures_reshaped = figures.view(figures.shape[0], figures.shape[1], -1)
                predicted_figures = torch.stack(predicted_figures, dim=1)
                predicted_figures_reshaped = predicted_figures.reshape(
                    predicted_figures.shape[0], predicted_figures.shape[1], -1
                )
                with torch.no_grad():
                    _, transposed_indices = hungarian_slots_loss(
                        figures_reshaped,
                        predicted_figures_reshaped,
                        device=device,
                    )
                avg_score, _ = r2_score(
                    mapped_latents, predicted_latents, transposed_indices
                )
                r2_scores.append(avg_score)
            id_scores.append(np.mean(r2_scores))

        if id_scores[0] > best_id_scores[0]:
            best_id_scores[0] = id_scores[0]
        if id_scores[1] > best_id_scores[1]:
            best_id_scores[1] = id_scores[1]

        print(f"Best OOD id score: {best_id_scores[1]}")
        print(f"Best ID id score: {best_id_scores[0]}")

    return best_id_scores[0].item(), best_id_scores[1].item()
