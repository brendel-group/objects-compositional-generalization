from typing import Tuple

import numpy as np
import sklearn
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchmetrics import R2Score

from .utils.data_utlis import EvalDataset
from .models.models_utils import MLP


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


def correlation(Z, hZ, num_slots, inf_slot_dim, gt_slot_dim, indices=None):
    corr_cont = np.zeros((num_slots, num_slots))
    corr_disc = np.zeros((num_slots, num_slots))

    hZ = hZ.view(len(hZ), num_slots, inf_slot_dim).permute(1, 0, 2).cpu()
    Z = Z.view(len(Z), num_slots, gt_slot_dim).permute(1, 0, 2).cpu()
    Z_cont = Z[:, :, 0:4]
    Z_disc = Z[:, :, 4]
    for i in range(num_slots):
        for j in range(num_slots):
            ZS_cont = Z_cont[i]
            ZS_disc = Z_disc[i]
            hZS = hZ[j]
            # if args.data != "synth":
            if j > i:
                hZS = hZS[indices[j]]
                ZS_cont = ZS_cont[indices[j]]
                ZS_disc = ZS_disc[indices[j]]
            elif i >= j:
                ZS_cont = ZS_cont[indices[i]]
                ZS_disc = ZS_disc[indices[i]]
                hZS = hZS[indices[i]]

            scaler_Z = sklearn.preprocessing.StandardScaler()
            scaler_hZ = sklearn.preprocessing.StandardScaler()
            z_train_cont, z_eval_cont = np.split(
                scaler_Z.fit_transform(ZS_cont), [int(0.8 * len(ZS_cont))]
            )
            z_train_disc, z_eval_disc = np.split(ZS_disc, [int(0.8 * len(ZS_disc))])
            hz_train, hz_eval = np.split(
                scaler_hZ.fit_transform(hZS), [int(0.8 * len(hZS))]
            )

            train_loader = DataLoader(
                EvalDataset(torch.from_numpy(hz_train), torch.from_numpy(z_train_cont)),
                batch_size=10,
            )
            corr_cont[i, j] = train_mlp_eval(
                train_loader,
                hz_eval,
                z_eval_cont,
                inf_slot_dim,
                gt_slot_dim - 1,
            )
            train_loader = DataLoader(
                EvalDataset(torch.from_numpy(hz_train), z_train_disc), batch_size=10
            )
            corr_disc[i, j] = train_mlp_eval(
                train_loader, hz_eval, z_eval_disc, inf_slot_dim, 3, discrete=True
            )
    return corr_cont, corr_disc


def train_mlp_eval(
    train_loader, hZ_val, Z_val, inp_dim, out_dim, discrete=False, device="cuda"
):
    net = MLP(inp_dim, out_dim, hidden_dim=256, n_layers=3, nonlinear=True).to(device)
    epoch = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)
    r2_best = 0.0
    while epoch < 40:
        epoch += 1
        net.train()
        if epoch == 25:
            optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)
        for x, y in train_loader:
            optimizer.zero_grad()
            yh = net(x.float().to(device))
            if discrete:
                loss_func = torch.nn.CrossEntropyLoss()
                y = y - 1
                loss = loss_func(yh, y.long().to(device))
            else:
                loss = (yh - y.float().to(device)).square().mean()
            loss.backward()
            optimizer.step()
        net.eval()

        with torch.no_grad():
            if discrete:
                hz_pred = net(torch.from_numpy(hZ_val).float().to(device)).cpu()
                y_pred_softmax = torch.log_softmax(hz_pred, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                correct_pred = (y_pred_tags == Z_val - 1).float()
                r2 = correct_pred.sum() / len(correct_pred)
                r2 = r2.item()
            else:
                hz_pred = net(torch.from_numpy(hZ_val).float().to(device)).cpu()
                r2 = sklearn.metrics.r2_score(torch.from_numpy(Z_val).float(), hz_pred)

            if r2_best < r2:
                r2_best = r2
    return r2_best
