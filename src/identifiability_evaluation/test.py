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

from typing import List, Optional, Sequence, Tuple

import numpy as np
import sklearn.metrics
import sklearn.preprocessing
import torch
from torch import nn

from . import matching


def _compute_test_performance_single_model(
    batched_z_test_output: List[np.ndarray],
    batched_z_test_gts: List[np.ndarray],
    losses: List[np.ndarray],
    categorical_dimensions: Sequence[int],
    scaler_z: Optional[sklearn.preprocessing.StandardScaler],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given predictions, ground-truths, and losses, compute the test performance.

    Args:
        z_test_output: List of arrays of shape (N, slot_dimensionality).
        z_test_gts: List of arrays of shape (N, slot_dimensionality).
        losses: List of arrays of shape (N,).
        categorical_dimensions: List of categorical dimensions.
        scaler_z: Scaler used to scale the continuous latent variables (optional).

    Returns:
        The R² score for continuous variables, the accuracy for categorical variables,
        and the total test loss.
    """

    z_test_output = np.concatenate(batched_z_test_output, 0)
    z_test_gts = np.concatenate(batched_z_test_gts, 0)

    loss = np.mean(losses)

    continuous_dimensions = [
        i for i in range(z_test_output.shape[-1]) if i not in categorical_dimensions
    ]

    z_test_output_cont = z_test_output[..., continuous_dimensions]
    z_test_output_cat = z_test_output[..., categorical_dimensions]

    z_test_gts_cont = z_test_gts[..., continuous_dimensions]
    z_test_gts_cat = z_test_gts[..., categorical_dimensions]

    if scaler_z is not None:
        z_test_output_cont = scaler_z.inverse_transform(z_test_output_cont)
        z_test_gts_cont = scaler_z.inverse_transform(z_test_gts_cont)

    r2 = sklearn.metrics.r2_score(z_test_gts_cont, z_test_output_cont)
    if len(categorical_dimensions) > 0:
        accuracy = np.mean(z_test_gts_cat == z_test_output_cat)
    else:
        accuracy = np.nan

    return r2, accuracy, loss


def test_ondiagonal_model(
    model_ondiagonal: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    categorical_dimensions: Sequence[int],
    scaler_z: Optional[sklearn.preprocessing.StandardScaler],
):
    """Test the on-diagonal model.

    Args:
        model_ondiagonal: Model to test.
        test_loader: Test loader.
        categorical_dimensions: List of categorical dimensions.
        scaler_z: Scaler used to scale the continuous latent variables (optional).

    Returns:
        The R² score for continuous variables, the accuracy for categorical variables,
        and the total test loss.
    """

    z_test_output = []
    z_test_gts = []
    losses = []
    with torch.no_grad():
        for z_pred, z, z_mask in test_loader:
            gt_slot_dimensionality = z.shape[-1]
            (
                losses_ondiagonal,
                _,
                indices_ondiagonal,
                preds_ondiagonal,
            ) = matching.get_matched_predictions(
                z_pred, z, categorical_dimensions, model_ondiagonal, target_mask=z_mask
            )
            losses.append(
                losses_ondiagonal[z_mask].mean().cpu().item() * gt_slot_dimensionality,
            )
            preds_ondiagonal = preds_ondiagonal[z_mask]
            z = z[z_mask]
            z_test_output.append(preds_ondiagonal.cpu().numpy())
            z_test_gts.append(z.cpu().numpy())

    return _compute_test_performance_single_model(
        z_test_output, z_test_gts, losses, categorical_dimensions, scaler_z
    )


def test_offdiagonal_model(
    model_offdiagonal: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    categorical_dimensions: Sequence[int],
    scaler_z: Optional[sklearn.preprocessing.StandardScaler],
):
    """Test the off-diagonal model.

    Args:
        model_offdiagonal: Model to test.
        test_loader: Test loader.
        categorical_dimensions: List of categorical dimensions.
        scaler_z: Scaler used to scale the continuous latent variables (optional).

    Returns:
        The R² score for continuous variables, the accuracy for categorical variables,
        and the total test loss.
    """

    z_test_output = []
    z_test_gts = []
    losses = []
    with torch.no_grad():
        for z_pred, z, z_mask, indices_ondiagonal in test_loader:
            gt_slot_dimensionality = z.shape[-1]
            (
                losses_offdiagonal,
                _,
                _,
                preds_offdiagonal,
            ) = matching.get_matched_predictions(
                z_pred,
                z,
                categorical_dimensions,
                model_offdiagonal,
                indices_ondiagonal,
                target_mask=z_mask,
            )
            losses.append(
                losses_offdiagonal[z_mask].mean().cpu().item() * gt_slot_dimensionality
            )
            preds_offdiagonal = preds_offdiagonal[z_mask]
            z = z[z_mask]
            z_test_output.append(preds_offdiagonal.cpu().numpy())
            z_test_gts.append(z.cpu().numpy())

    return _compute_test_performance_single_model(
        z_test_output, z_test_gts, losses, categorical_dimensions, scaler_z
    )


def test_models(
    model_ondiagonal: nn.Module,
    model_offdiagonal: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    categorical_dimensions: Sequence[int],
    scaler_z: Optional[sklearn.preprocessing.StandardScaler],
    verbose: int = 0,
):
    """Test the on-diagonal and off-diagonal models.

    Args:
        model_ondiagonal: On-diagonal model to test.
        model_offdiagonal: Off-diagonal model to test.
        test_loader: Test loader.
        categorical_dimensions: List of categorical dimensions.
        scaler_z: Scaler used to scale the continuous latent variables (optional).
        verbose: Verbosity level.

    Returns:
        The R² score for continuous variables, the accuracy for categorical variables,
        and the total test loss. Each of these is a 2-tuple containing the scores for
        the on- and off-diagonal model, respectively.
    """

    batched_z_test_output = []
    batched_z_test_gts = []
    batched_losses = []

    with torch.no_grad():
        for z_pred, z, z_mask in test_loader:
            gt_slot_dimensionality = z.shape[-1]
            (
                losses_ondiagonal,
                _,
                indices_ondiagonal,
                preds_ondiagonal,
            ) = matching.get_matched_predictions(
                z_pred, z, categorical_dimensions, model_ondiagonal, target_mask=z_mask
            )
            indices_ondiagonal = torch.stack(
                (
                    indices_ondiagonal[:, 0][z_mask[:, 0]],
                    indices_ondiagonal[:, 1][z_mask[:, 0]],
                ),
                -1,
            )
            (
                losses_offdiagonal,
                _,
                _,
                preds_offdiagonal,
            ) = matching.get_matched_predictions(
                z_pred,
                z,
                categorical_dimensions,
                model_offdiagonal,
                indices_ondiagonal,
                target_mask=z_mask,
            )
            batched_losses.append(
                (
                    losses_ondiagonal[z_mask].mean().cpu().item()
                    * gt_slot_dimensionality,
                    losses_offdiagonal[z_mask].mean().cpu().item()
                    * gt_slot_dimensionality,
                )
            )
            preds_ondiagonal = preds_ondiagonal[z_mask]
            preds_offdiagonal = preds_offdiagonal[z_mask]
            z = z[z_mask]
            batched_z_test_output.append(
                np.stack(
                    (preds_ondiagonal.cpu().numpy(), preds_offdiagonal.cpu().numpy()), 0
                )
            )
            batched_z_test_gts.append(z.cpu().numpy())
    # z_test_output.shape: (2, n_test_samples*n_masked_slots, slot_dimensionality)
    z_test_output = np.concatenate(batched_z_test_output, 1)
    # z_test_gts.shape: (n_test_samples*n_masked_slots, slot_dimensionality)
    z_test_gts = np.concatenate(batched_z_test_gts, 0)

    continuous_dimensions = [
        i for i in range(z_test_output.shape[-1]) if i not in categorical_dimensions
    ]

    gt_slot_dimensionality_cont = len(continuous_dimensions)

    losses = np.mean(np.stack(batched_losses, 0), 0)

    z_test_output_cont = z_test_output[..., continuous_dimensions]
    z_test_output_cat = z_test_output[..., categorical_dimensions]

    z_test_gts_cont = z_test_gts[..., continuous_dimensions]
    z_test_gts_cat = z_test_gts[..., categorical_dimensions]

    if scaler_z is not None:
        z_test_output_cont = scaler_z.inverse_transform(
            z_test_output_cont.reshape((-1, gt_slot_dimensionality_cont))
        ).reshape(z_test_output_cont.shape)
        z_test_gts_cont = scaler_z.inverse_transform(z_test_gts_cont)

    r2_ondiagonal = sklearn.metrics.r2_score(z_test_gts_cont, z_test_output_cont[0])
    r2_offdiagonal = sklearn.metrics.r2_score(z_test_gts_cont, z_test_output_cont[1])

    if len(categorical_dimensions) > 0:
        accuracy_ondiagonal = np.mean(z_test_gts_cat == z_test_output_cat[0])
        accuracy_offdiagonal = np.mean(z_test_gts_cat == z_test_output_cat[1])
    else:
        accuracy_ondiagonal = np.nan
        accuracy_offdiagonal = np.nan

    r2_scores = (r2_ondiagonal, r2_offdiagonal)
    accuracies = (accuracy_ondiagonal, accuracy_offdiagonal)

    return r2_scores, accuracies, losses
