# Copyright 2023 Roland S. Zimmermann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F


def hungarian_algorithm(
    cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes
    the overall cost. Returns the matching indices as a LongTensor with shape
    (batch size, 2, min(num objects, num slots)). The first column is the row
    indices (the indices of the true objects) while the second column is the
    column indices (the indices of the slots). The row indices are always in
    ascending order, while the column indices are not necessarily. The outputs
    are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the
    row indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots))
                with the costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects,
                num slots)) containing the indices for the resulting matching.

    From https://github.com/addtt/object-centric-library/utils/slot_matching.py.
    """

    # List of tuples of size 2 containing flat arrays
    raw_indices = list(
        map(scipy.optimize.linear_sum_assignment, cost_matrix.cpu().detach().numpy())
    )
    indices = torch.tensor(
        np.array(raw_indices), device=cost_matrix.device, dtype=torch.long
    )
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    return smallest_cost_matrix.to(cost_matrix.device), indices


def stochastic_hungarian_algorithm(
    cost_matrix: torch.Tensor, relative_perturbation_strength: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Noisy version of hungarian_algorithm, that samples from a noisy process.

    Adds normally distributed noise to the cost matrix (independently for each
    row) and then samples a single solution from this noisy cost matrix.

    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
        relative_perturbation_strength: How large the added noise should be
            relative to the cost values per row.
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots))
                with the costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects,
                num slots)) containing the indices for the resulting matching.
    """

    cost_matrix_np = cost_matrix.cpu().detach().numpy()
    cost_matrix_masked = cost_matrix_np.copy()
    cost_matrix_masked[np.isinf(cost_matrix_masked)] = 0.0
    relative_variance = (
        relative_perturbation_strength
        * np.max(cost_matrix_masked, axis=-1, keepdims=True)
        * scipy.special.softmax(cost_matrix_masked, axis=-1)
    )

    noise = relative_variance * np.random.randn(*cost_matrix.shape)
    stochastic_cost_matrix = cost_matrix_np + noise

    # List of tuples of size 2 containing flat arrays
    raw_indices = list(
        map(scipy.optimize.linear_sum_assignment, stochastic_cost_matrix)
    )

    indices = torch.tensor(
        np.array(raw_indices), device=cost_matrix.device, dtype=torch.long
    )
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    return smallest_cost_matrix.to(cost_matrix.device), indices


def categorical_loss(
    label: torch.Tensor,
    logits: torch.Tensor,
    mode: Union[Literal["train"], Literal["test"]],
):
    """Computes the loss for a categorical variable.

    Args:
        label: Tensor of shape (batch size, num slots) containing the labels.
        logits: Tensor of shape (batch size, num slots, num classes) containing
            the predicted logits.
        mode: Either "train" or "test". In "train" mode, the loss is computed
            using cross-entropy. In "test" mode, the loss is computed using
            the max-margin loss (which has similar value range to MSE of continuous
            factors of variation).
    Returns:
        A Tensor with shape (batch size, num slots, num slots) containing the loss for
        each possible pairing of slots.
    """
    assert logits.shape[1] == 1
    logits = logits[:, 0]
    label = label.long()

    if mode == "test":
        p = torch.softmax(logits, -1)
        max_p, _ = torch.max(p, -1)
        target_p = p[
            torch.arange(p.shape[0])[:, None], torch.arange(p.shape[1])[None], label
        ]
        losses = max_p.unsqueeze(-2) - target_p.unsqueeze(-1)
        losses = torch.clamp(losses, min=0)
    elif mode == "train":
        # Transform logits to shape (BS, #Classes, #Slots, #Slots)
        # Transform labels to shape (BS, #Slots, #Slots)
        losses = F.cross_entropy(
            torch.tile(
                torch.transpose(logits, -1, -2).unsqueeze(2), (1, 1, label.shape[-1], 1)
            ),
            torch.tile(label.unsqueeze(2), (1, 1, label.shape[-1])),
            reduction="none",
        )

    return losses


def get_matched_predictions(
    x: torch.Tensor,
    target: torch.Tensor,
    categorical_dimensions: Sequence[int],
    model: nn.Module,
    masking_indices: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    slack: float = 0.0,
):
    """Obtains the optimal predictions for a model for an unclear matching.

    Args:
        x: The input data of the model.
        target: The target data.
        categorical_dimensions: List of target dimensions containing
            categorical variables.
        model: The model.
        masking_indices: If given, these indices are marked as non-feasible
            solutions in the matching process such that those get ignored.
        target_mask: Boolean mask indicating which target slots to use and
            which to ignore, e.g., because they were just added as placeholders
            when converting a sparse to a dense tensor.
        slack: Controls the slack given to the (stochastic) matching process.

    Returns:
        Tuple of (1) tensor containing the optimal loss values, (2) the indices
        determined by the matching procedure and (3) the optimal predictions.
    """

    continuous_dimensions = [
        i for i in range(target.shape[-1]) if i not in categorical_dimensions
    ]
    target_cat = target[..., categorical_dimensions]
    target_cont = target[..., continuous_dimensions]
    preds_cont, preds_cat = model(x)
    loss_train = torch.zeros(target.shape[0], target.shape[1], target.shape[1]).to(
        target.device
    )
    loss_test = torch.zeros_like(loss_train)
    if preds_cont is not None:
        if preds_cont.ndim == 3:
            unflatten = True
            preds_cont = preds_cont.unsqueeze(1)
        else:
            unflatten = False
        loss_cont = ((target_cont.unsqueeze(2) - preds_cont) ** 2).sum(-1)

        if target_mask is not None:
            loss_cont[~target_mask] = 200 * torch.max(loss_cont).detach()

        loss_train = loss_train + loss_cont
        loss_test = loss_test + loss_cont

    if preds_cat is not None:
        if preds_cat[0].ndim == 3:
            unflatten = True
            preds_cat = [p.unsqueeze(1) for p in preds_cat]
        else:
            unflatten = False
        losses_cat_train_per_dimension = []
        losses_cat_test_per_dimension = []
        for i in range(len(preds_cat)):
            losses_cat_train_per_dimension.append(
                categorical_loss(target_cat[..., i], preds_cat[i], "train")
            )
            losses_cat_test_per_dimension.append(
                categorical_loss(target_cat[..., i], preds_cat[i], "test")
            )
        losses_cat_train = torch.sum(torch.stack(losses_cat_train_per_dimension, 0), 0)
        losses_cat_test = torch.sum(torch.stack(losses_cat_test_per_dimension, 0), 0)

        if target_mask is not None:
            losses_cat_train[~target_mask] = 200 * torch.max(losses_cat_train).detach()
            losses_cat_test[~target_mask] = 200 * torch.max(losses_cat_test).detach()

        loss_train = loss_train + losses_cat_train
        loss_test = loss_test + losses_cat_test

        preds_cat = torch.stack([torch.argmax(p, -1) for p in preds_cat], -1)

    if masking_indices is not None:
        loss_test[
            np.arange(len(loss_test))[:, None],
            masking_indices[:, 0],
            masking_indices[:, 1],
        ] = np.inf
    if slack > 0:
        optimal_loss, optimal_indices = stochastic_hungarian_algorithm(loss_test, 0.2)
    else:
        optimal_loss, optimal_indices = hungarian_algorithm(loss_test)

    optimal_train_loss = torch.stack(
        [
            loss_train[i][optimal_indices[i, 0], optimal_indices[i, 1]]
            for i in range(loss_train.shape[0])
        ]
    )

    if unflatten:
        if preds_cont is not None:
            optimal_preds_cont = preds_cont[:, 0][
                np.arange(len(loss_test))[:, None], optimal_indices[:, 1]
            ]
        else:
            optimal_preds_cont = None
        if preds_cat is not None:
            optimal_preds_cat = preds_cat[:, 0][
                np.arange(len(loss_test))[:, None], optimal_indices[:, 1]
            ]
        else:
            optimal_preds_cat = None
    else:
        if preds_cont is not None:
            optimal_preds_cont = preds_cont[
                np.arange(len(loss_test))[:, None],
                optimal_indices[:, 0],
                optimal_indices[:, 1],
            ]
        else:
            optimal_preds_cont = None

        if preds_cat is not None:
            optimal_preds_cat = preds_cat[
                np.arange(len(loss_test))[:, None],
                optimal_indices[:, 0],
                optimal_indices[:, 1],
            ]
        else:
            optimal_preds_cat = None

    if optimal_preds_cont is not None and optimal_preds_cat is not None:
        optimal_preds = torch.zeros(
            (
                *optimal_preds_cont.shape[:-1],
                optimal_preds_cont.shape[-1] + optimal_preds_cat.shape[-1],
            )
        ).to(x.device)
        optimal_preds[..., categorical_dimensions] = optimal_preds_cat.float()
        optimal_preds[..., continuous_dimensions] = optimal_preds_cont
    elif optimal_preds_cont is not None:
        optimal_preds = optimal_preds_cont
    elif optimal_preds_cat is not None:
        optimal_preds = optimal_preds_cat

    return optimal_loss, optimal_train_loss, optimal_indices, optimal_preds
