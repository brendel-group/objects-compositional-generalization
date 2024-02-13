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
import torch
from torch import nn

from . import datasets, matching, test


def check_training_stop_reached(
    epoch: int,
    train_losses: List[torch.Tensor],
    val_losses: List[torch.Tensor],
    relative_stopping_threshold: float,
    absolute_stopping_threshold: float = np.inf,
    relative_val_stopping_threshold: float = 0.1,
    n_stopping_epochs: int = 7,
    max_training_epochs: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """Returns True of early stopping criteria are matched.

    Args:
        epoch: Index of current epoch.
        train_losses: Trajectory of train losses.
        val_losses: Trajectory of validation losses.
        relative_stopping_threshold: Once all relative loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        absolute_stopping_threshold: Once all absolute loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        relative_val_stopping_threshold: How much the validation loss can be
            increased relative to the currently known minimum before the
            training terminates.
        n_stopping_epochs: Number of epochs to look back for checking the stopping
            criteria.
        max_training_epochs: After how many epochs the training will be stopped.

    Returns:
        Tuple containing a bool and an optional string. If the bool is True, one of the
        early stopping criteria is matched; then, the string contains the name of the
        criterion.
    """

    if max_training_epochs is not None and epoch == max_training_epochs:
        return True, "max epochs"

    if len(val_losses) > 1:
        if val_losses[-1] > (1 + relative_val_stopping_threshold) * np.min(val_losses):
            return True, "validation"

    if len(train_losses) > n_stopping_epochs + 1:
        masked_train_losses = np.array(train_losses)
        masked_train_losses[np.isinf(masked_train_losses)] = 0
        delta_train_losses = 2 * (np.roll(masked_train_losses, 1) - masked_train_losses)
        relative_delta_train_losses = (
            2
            * (np.roll(masked_train_losses, 1) - masked_train_losses)
            / (np.roll(masked_train_losses, 1) + masked_train_losses)
        )
        relative_condition_reached = np.all(
            relative_delta_train_losses[-n_stopping_epochs:]
            < relative_stopping_threshold
        )
        absolute_condition_reached = np.all(
            delta_train_losses[-n_stopping_epochs:] < absolute_stopping_threshold
        )

        if (relative_stopping_threshold == np.inf or relative_condition_reached) and (
            absolute_stopping_threshold == np.inf or absolute_condition_reached
        ):
            status = ""
            if relative_condition_reached:
                status += "Relative loss difference is below {0} ({1}). ".format(
                    relative_stopping_threshold, relative_delta_train_losses[-1]
                )
            if absolute_condition_reached:
                status += "Absolute loss difference is below {0} ({1}). ".format(
                    absolute_stopping_threshold, delta_train_losses[-1]
                )

            print("Stopping optimization in epoch {0}. {1}".format(epoch, status))
            return True, "train"

    return False, None


def train_ondiagonal_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    categorical_dimensions: Sequence[int],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    relative_stopping_threshold: float,
    absolute_stopping_threshold: float = np.inf,
    relative_val_stopping_threshold: float = 0.1,
    max_training_epochs: Optional[int] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    matching_slack: float = 0.1,
    verbose: int = 0,
):
    """Train the on-/off-diagonal models between ground-truth and target data.

    Args:
        train_loader: Dataloader for training.
        val_loader: Dataloader for determining early stopping.
        categorical_dimensions: List of ground-truth dimensions containing
            categorical variables.
        model: Model for on-diagonal readouts.
        optimizer: Optimizer for training on-diagonal readouts.
        relative_stopping_threshold: Once all relative loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        absolute_stopping_threshold: Once all absolute loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        relative_val_stopping_threshold: How much the validation loss can be
            increased relative to the currently known minimum before the
            training terminates.
        max_training_epochs: After how many epochs the training will be stopped.
        lr_scheduler: Learning rate scheduler.
        matching_slack: Slack coefficient for the matching; the larger this
            value is, the more noisy the matching becomes which can help
            avoiding local minima.
        verbose: Controls the verbosity level.
    """

    if absolute_stopping_threshold == np.inf and relative_stopping_threshold == np.inf:
        raise ValueError(
            "Either relative_stopping_threshold or "
            "absolute_stopping_threshold must be finite."
        )

    if max_training_epochs == 0:
        return

    train_losses: List[torch.Tensor] = []
    val_losses: List[torch.Tensor] = []
    epoch = 0
    checkpoint = model.state_dict()
    while True:
        epoch += 1
        current_epoch_train_losses = []
        for i, (z_pred, z, z_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            _, losses, indices, _ = matching.get_matched_predictions(
                z_pred,
                z,
                categorical_dimensions,
                model,
                slack=matching_slack,
                target_mask=z_mask,
            )
            indices = torch.stack(
                (indices[:, 0][z_mask[:, 0]], indices[:, 1][z_mask[:, 0]]), -1
            )
            # loss = torch.mean(losses[z_mask].sum(0), 0)
            loss = losses[z_mask].mean() * losses.shape[1]
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            current_epoch_train_losses.append(loss.item())
        current_epoch_avg_train_loss = np.mean(current_epoch_train_losses, 0)
        train_losses.append(current_epoch_avg_train_loss)

        _, _, val_loss = test.test_ondiagonal_model(
            model, val_loader, categorical_dimensions, None
        )
        if len(val_losses) > 0 and val_loss < np.min(val_losses):
            checkpoint = model.state_dict()
        val_losses.append(val_loss)

        if verbose > 0:
            print(
                f"Epoch: {epoch}. Train Loss: {current_epoch_avg_train_loss} "
                f"Val Loss: {val_loss}"
            )

        abort, abort_reason = check_training_stop_reached(
            epoch,
            train_losses,
            val_losses,
            relative_stopping_threshold,
            absolute_stopping_threshold,
            relative_val_stopping_threshold,
            max_training_epochs=max_training_epochs,
        )

        if abort:
            if abort_reason == "validation":
                # Restore previous checkpoint since overfitting was detected.
                model.load_state_dict(checkpoint)
            break


def train_offdiagonal_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    categorical_dimensions: Sequence[int],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    relative_stopping_threshold: float,
    absolute_stopping_threshold: float = np.inf,
    relative_val_stopping_threshold: float = 0.1,
    max_training_epochs: Optional[int] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    matching_slack: float = 0.1,
    verbose: int = 0,
):
    """Fit and evaluate on-/off-diagonal models between z and z_pred.

    Args:
        train_loader: Dataloader for training.
        val_loader: Dataloader for determining early stopping.
        categorical_dimensions: List of ground-truth dimensions containing
            categorical variables.
        model: Model for off-diagonal readouts.
        optimizer: Optimizer for training off-diagonal readouts.
        relative_stopping_threshold: Once all relative loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        absolute_stopping_threshold: Once all absolute loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        relative_val_stopping_threshold: How much the validation loss can be
            increased relative to the currently known minimum before the
            training terminates.
        max_training_epochs: After how many epochs the training will be stopped.
        lr_scheduler: Learning rate scheduler.
        matching_slack: Slack coefficient for the matching; the larger this
            value is, the more noisy the matching becomes which can help
            avoiding local minima.
        verbose: Controls the verbosity level.
    """

    if absolute_stopping_threshold == np.inf and relative_stopping_threshold == np.inf:
        raise ValueError(
            "Either relative_stopping_threshold or "
            "absolute_stopping_threshold must be finite."
        )

    if max_training_epochs == 0:
        return

    train_losses: List[torch.Tensor] = []
    val_losses: List[torch.Tensor] = []
    epoch = 0
    checkpoint = model.state_dict()
    while True:
        epoch += 1
        current_epoch_train_losses = []

        for i, (z_pred, z, z_mask, indices_ondiagonal) in enumerate(train_loader):
            optimizer.zero_grad()
            _, losses, indices_offdiagonal, _ = matching.get_matched_predictions(
                z_pred,
                z,
                categorical_dimensions,
                model,
                indices_ondiagonal,
                slack=matching_slack,
                target_mask=z_mask,
            )
            # Ignore nans that might get introduced by masking.
            # loss_offdiagonal = torch.nanmean(loss_offdiagonal.sum(-1))
            # loss = torch.mean(losses[z_mask].sum(0), 0)
            loss = losses[z_mask].mean() * losses.shape[1]
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            current_epoch_train_losses.append(loss.item())
        current_epoch_avg_train_loss = np.mean(current_epoch_train_losses, 0)
        train_losses.append(current_epoch_avg_train_loss)

        _, _, val_loss = test.test_offdiagonal_model(
            model, val_loader, categorical_dimensions, None
        )
        if len(val_losses) > 0 and val_loss < np.min(val_losses):
            checkpoint = model.state_dict()
        val_losses.append(val_loss)

        if verbose > 0:
            print(
                f"Epoch: {epoch}. Train Loss: {current_epoch_avg_train_loss} "
                f"Val Loss: {val_loss}"
            )

        abort, abort_reason = check_training_stop_reached(
            epoch,
            train_losses,
            val_losses,
            relative_stopping_threshold,
            absolute_stopping_threshold,
            relative_val_stopping_threshold,
            max_training_epochs=max_training_epochs,
        )

        if abort:
            if abort_reason == "validation":
                # Restore previous checkpoint since overfitting was detected.
                model.load_state_dict(checkpoint)
            break


def train_models(
    z_pred: torch.Tensor,
    z: torch.Tensor,
    z_mask: torch.Tensor,
    categorical_dimensions: Sequence[int],
    val_loader: torch.utils.data.DataLoader,
    model_ondiagonal: nn.Module,
    model_offdiagonal: nn.Module,
    optimizer_ondiagonal: torch.optim.Optimizer,
    optimizer_offdiagonal: torch.optim.Optimizer,
    batch_size: int,
    relative_stopping_threshold: float,
    relative_val_stopping_threshold: float = 0.1,
    lr_scheduler_ondiagonal: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    lr_scheduler_offdiagonal: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    absolute_stopping_threshold: float = np.inf,
    max_training_epochs: Optional[int] = None,
    matching_slack: float = 0.1,
    augmentation_gaussian_std: float = 0.005,
    verbose: int = 0,
):
    """
    Args:
        z: Tensor containing the ground-truth data.
        z_pred: Tensor containing the predicted data.
        z_mask: Boolean mask indicating which ground-truth slots to use and
            which to ignore, e.g., because they were just added as placeholders
            when converting a sparse to a dense tensor.
        categorical_dimensions: List of ground-truth dimensions containing
            categorical variables.
        val_loader: Validation data loader.
        model_ondiagonal: Model for on-diagonal readouts.
        model_offdiagonal: Model for off-diagonal readouts.
        optimizer_ondiagonal: Optimizer for training on-diagonal readouts.
        optimizer_offdiagonal: Optimizer for training off-diagonal readouts.
        batch_size: Batch size for training.
        relative_stopping_threshold: Once all relative loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        relative_val_stopping_threshold: How much the validation loss can be
            increased relative to the currently known minimum before the
            training terminates.
        lr_scheduler_ondiagonal: Learning rate scheduler for ondiagonal model.
        lr_scheduler_offdiagonal: Learning rate scheduler for offdiagonal model.
        absolute_stopping_threshold: Once all absolute loss changes are less
            than this threshold, training stops. Both threshold conditions
            need to be satisfied for the optimization to be stopped. At least
            one needs to be set to a finite value.
        max_training_epochs: After how many epochs the training will be stopped.
        matching_slack: Slack coefficient for the matching; the larger this
            value is, the more noisy the matching becomes which can help
            avoiding local minima.
        augmentation_gaussian_std: Standard deviation of the Gaussian noise
            that is added to the input data to avoid/reduce overfitting.
        verbose: Controls the verbosity level.
    """

    # Augment input data with low-magnitude Gaussian noise to avoid/reduce
    # overfitting.
    def _augment_data(x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return (
            x[0] + torch.randn_like(x[0]) * augmentation_gaussian_std,
            *[it for it in x[1:]],
        )

    train_loader_ondiagonal = torch.utils.data.DataLoader(
        datasets.TransformableTensorDataset((z_pred, z, z_mask), _augment_data),
        batch_size=batch_size,
        shuffle=True,
    )

    train_loader_ondiagonal_clean = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_pred, z, z_mask),
        batch_size=batch_size,
        shuffle=True,
    )

    if verbose > 0:
        print("Fitting on-diagonal model.")
    model_ondiagonal.train()
    train_ondiagonal_model(
        train_loader_ondiagonal,
        val_loader,
        categorical_dimensions,
        model_ondiagonal,
        optimizer_ondiagonal,
        relative_stopping_threshold,
        absolute_stopping_threshold,
        relative_val_stopping_threshold,
        max_training_epochs,
        lr_scheduler_ondiagonal,
        matching_slack,
        verbose=verbose - 1,
    )

    # Construct dataloader for training the off-diagonal model(s).
    def get_offdiagonal_loader(loader: torch.utils.data.DataLoader):
        data_offdiagonal = []
        with torch.no_grad():
            for z_pred, z, z_mask in loader:
                _, _, indices_ondiagonal, _ = matching.get_matched_predictions(
                    z_pred,
                    z,
                    categorical_dimensions,
                    model_ondiagonal,
                    target_mask=z_mask,
                )
                data_offdiagonal.append((z_pred, z, z_mask, indices_ondiagonal))
        loader_offdiagonal = torch.utils.data.DataLoader(
            datasets.TransformableTensorDataset(
                [torch.cat([it[i] for it in data_offdiagonal]) for i in range(4)],
                _augment_data,
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        return loader_offdiagonal

    train_loader_offdiagonal = get_offdiagonal_loader(train_loader_ondiagonal_clean)
    val_loader_offdiagonal = get_offdiagonal_loader(val_loader)

    if verbose > 0:
        print("Fitting off-diagonal model.")
    model_offdiagonal.train()
    train_offdiagonal_model(
        train_loader_offdiagonal,
        val_loader_offdiagonal,
        categorical_dimensions,
        model_offdiagonal,
        optimizer_offdiagonal,
        relative_stopping_threshold,
        absolute_stopping_threshold,
        relative_val_stopping_threshold,
        max_training_epochs,
        lr_scheduler_offdiagonal,
        matching_slack,
        verbose=verbose - 1,
    )
