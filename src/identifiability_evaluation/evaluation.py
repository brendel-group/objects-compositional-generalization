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

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import sklearn.preprocessing
import torch
from lion_pytorch import Lion
from torch import nn

from . import models, test, train

__all__ = ["evaluate_model"]


def _internal_evaluate_model(
    z: torch.Tensor,
    z_pred: torch.Tensor,
    categorical_dimensions: Sequence[int],
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    model_depth: int = 1,
    relative_stopping_threshold: float = 0.0001,
    absolute_stopping_threshold: float = np.inf,
    relative_val_stopping_threshold: float = 0.1,
    max_training_epochs: Optional[int] = None,
    batch_size: int = 2048,
    standard_scale: bool = True,
    matching_slack: float = 0.1,
    training_augmentation_gaussian_std: float = 0.005,
    z_mask_values: Optional[Union[torch.Tensor, float, int]] = None,
    verbose: int = 0,
    provided_models: nn.Module = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Evaluates a model by measuring predictability of the ground-truth.

    Args:
        z: Tensor containing the ground-truth data.
        z_pred: Tensor containing the predicted data.
        categorical_dimensions: List of ground-truth dimensions containing
            categorical variables.
        train_val_test_split: Tuple containing the relative size of training,
            validation and test split.
        model_depth: Number of layers of the readout model(s).
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
        batch_size: Batch size used during training; this should not be chosen
            to large to avoid getting stuck in local minima.
        standard_scale: Standardize input and output data before fitting the
            regression model(s).
        matching_slack: Slack coefficient for the matching; the larger this
            value is, the more noisy the matching becomes which can help
            avoiding local minima.
        training_augmentation_gaussian_std: Standard deviation of the Gaussian
            noise used for data augmentation during training to avoid/reduce
            overfitting.
        z_mask_values: Values in z that should be masked out. If None, no
            masking is performed.
        verbose: Controls the verbosity level.

    Returns:
        Tuple containing the R² scores and accuracies, where each element is a tuple
        of the form (on-diagonal performance, off-diagonal performance).
    """

    if not sum(train_val_test_split) == 1:
        raise ValueError("Values in train_val_test split need to add up to 1.")

    device = z.device

    n_samples, gt_n_slots, gt_slot_dimensionality = z.shape

    continuous_dimensions = [
        i for i in range(gt_slot_dimensionality) if i not in categorical_dimensions
    ]

    gt_cat_slot_dimensionality = len(categorical_dimensions)
    gt_cont_slot_dimensionality = len(continuous_dimensions)

    if len(categorical_dimensions) > 0:
        gt_num_categories = (
            torch.max(
                z[..., categorical_dimensions].view(-1, gt_cat_slot_dimensionality), 0
            )[0]
            .cpu()
            .numpy()
            .astype(int)
            + 1
        ).tolist()
    else:
        gt_num_categories = None

    assert gt_slot_dimensionality == (
        gt_cat_slot_dimensionality + gt_cont_slot_dimensionality
    )

    _, pred_n_slots, pred_slot_dimensionality = z_pred.shape
    n_train_samples = int(n_samples * train_val_test_split[0])
    n_val_samples = int(n_samples * train_val_test_split[1])
    z_train = z[:n_train_samples].cpu().detach().numpy()
    z_pred_train = z_pred[:n_train_samples].cpu().detach().numpy()
    z_val_test = z[n_train_samples:].cpu().detach().numpy()
    z_pred_val_test = z_pred[n_train_samples:].cpu().detach().numpy()

    if standard_scale:
        scaler_z = sklearn.preprocessing.StandardScaler().fit(
            z_train[..., continuous_dimensions].reshape(-1, gt_cont_slot_dimensionality)
        )
        scaler_z_pred = sklearn.preprocessing.StandardScaler().fit(
            z_pred_train.reshape(-1, pred_slot_dimensionality)
        )
        z_train[..., continuous_dimensions] = scaler_z.transform(
            z_train[..., continuous_dimensions].reshape(-1, gt_cont_slot_dimensionality)
        ).reshape(-1, gt_n_slots, gt_cont_slot_dimensionality)
        z_pred_train = scaler_z_pred.transform(
            z_pred_train.reshape(-1, pred_slot_dimensionality)
        ).reshape(-1, pred_n_slots, pred_slot_dimensionality)
        z_val_test[..., continuous_dimensions] = scaler_z.transform(
            z_val_test[..., continuous_dimensions].reshape(
                -1, gt_cont_slot_dimensionality
            )
        ).reshape(-1, gt_n_slots, gt_cont_slot_dimensionality)
        z_pred_val_test = scaler_z_pred.transform(
            z_pred_val_test.reshape(-1, pred_slot_dimensionality)
        ).reshape(-1, pred_n_slots, pred_slot_dimensionality)

    z_train = torch.Tensor(z_train).to(device)
    z_pred_train = torch.Tensor(z_pred_train).to(device)
    z_val = torch.Tensor(z_val_test[:n_val_samples]).to(device)
    z_pred_val = torch.Tensor(z_pred_val_test[:n_val_samples]).to(device)
    z_test = torch.Tensor(z_val_test[n_val_samples:]).to(device)
    z_pred_test = torch.Tensor(z_pred_val_test[n_val_samples:]).to(device)

    if z_mask_values is not None:
        if isinstance(z_mask_values, (int, float)):
            z_mask_values = torch.ones(z.shape[-1], device=device) * z_mask_values
        z_mask = ~torch.all(z == z_mask_values.view((1, 1, -1)), -1).to(device)
    else:
        z_mask = torch.ones(z.shape[:2], dtype=torch.bool, device=device)
    z_mask_train = z_mask[:n_train_samples]
    z_mask_val = z_mask[n_train_samples : n_train_samples + n_val_samples]
    z_mask_test = z_mask[n_train_samples + n_val_samples :]

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_pred_val, z_val, z_mask_val),
        batch_size=batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_pred_test, z_test, z_mask_test),
        batch_size=batch_size,
    )

    def cont_slot_readout_fn(
        n_input, n_output: int, hidden_dim: int = 256
    ) -> nn.Module:
        modules: List[nn.Module] = []
        n = n_input
        for _ in range(model_depth - 1):
            modules.append(nn.Linear(n, hidden_dim))
            modules.append(nn.LeakyReLU())
            n = hidden_dim
        modules.append(nn.Linear(n, n_output))
        return nn.Sequential(*modules)

    def cat_slot_readout_fn(n_input, n_output: int, hidden_dim: int = 256) -> nn.Module:
        return cont_slot_readout_fn(n_input, n_output, hidden_dim)

    optimizer_cls = Lion

    model_ondiagonal = models.Model(
        pred_n_slots,
        pred_slot_dimensionality,
        gt_cont_slot_dimensionality,
        cont_slot_readout_fn,
        gt_num_categories,
        cat_slot_readout_fn,
        share_readout=True,
    )
    model_ondiagonal = model_ondiagonal.to(z.device)
    optimizer_ondiagonal = optimizer_cls(
        model_ondiagonal.parameters(),
        lr=1e-4,
    )  # weight_decay=1e-3)

    model_offdiagonal = models.Model(
        pred_n_slots,
        pred_slot_dimensionality,
        gt_cont_slot_dimensionality,
        cont_slot_readout_fn,
        gt_num_categories,
        cat_slot_readout_fn,
        share_readout=True,
    )
    model_offdiagonal = model_offdiagonal.to(z.device)
    optimizer_offdiagonal = optimizer_cls(
        model_offdiagonal.parameters(),
        lr=1e-4,
    )  # weight_decay=1e-3)

    n_train_batches = max(len(z_train) // batch_size, 1)

    def get_lr_scheduler(optimizer):
        milestones = [n_train_batches * 200]
        schedulers = [
            torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=n_train_batches * max_training_epochs
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.1 ** (1 / (50 * n_train_batches)),
            ),
        ]
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones)

    if max_training_epochs is not None:
        lr_scheduler_ondiagonal = get_lr_scheduler(optimizer_ondiagonal)
        lr_scheduler_offdiagonal = get_lr_scheduler(optimizer_offdiagonal)
    else:
        lr_scheduler_ondiagonal = None
        lr_scheduler_offdiagonal = None

    if provided_models is None:
        train.train_models(
            z_pred_train,
            z_train,
            z_mask_train,
            categorical_dimensions,
            val_loader,
            model_ondiagonal,
            model_offdiagonal,
            optimizer_ondiagonal,
            optimizer_offdiagonal,
            batch_size,
            relative_stopping_threshold,
            relative_val_stopping_threshold,
            lr_scheduler_ondiagonal,
            lr_scheduler_offdiagonal,
            absolute_stopping_threshold,
            max_training_epochs,
            matching_slack,
            training_augmentation_gaussian_std,
            verbose,
        )
    else:
        model_ondiagonal = provided_models[0]
        model_offdiagonal = provided_models[1]

    model_ondiagonal.eval()
    model_offdiagonal.eval()

    r2_scores, accuracies, test_loss = test.test_models(
        model_ondiagonal,
        model_offdiagonal,
        test_loader,
        categorical_dimensions,
        scaler_z if standard_scale else None,
        verbose,
    )

    return r2_scores, accuracies, (model_ondiagonal, model_offdiagonal)


def compute_relative_performance(
    scores: Tuple[float, float], baseline_scores: Tuple[float, float]
) -> float:
    """Computes the relative information gap between (non-)matching slots.

    If r1 and r2 denote the scores (e.g, R² or accuracy) between the best pair of
    predicted and ground-truth slots and the second best pair, and rb denotes a
    0-1 clipped version of the latter but for the ground-truth data, then this function
    computes:

    score = (r1 - rb) / (1 - rb) - clip((r2 - rb) / (1 - rb), 0, 1).

    Args:
        scores: Tuple containing the score of the on-diagonal and the off-
        diagonal models/matchings for the object-centric model in question.
        baseline_scores: Tuple containing the score of the on-diagonal and
        the off- diagonal models/matchings computed just on the ground-truth
        data.

    Returns:
        Score as computed by equation shown above.
    """
    clipped_baseline_off_score = np.clip(baseline_scores[1], 0, 1 - 1e-6)
    normalizer = 1 - clipped_baseline_off_score
    score = (scores[0] - clipped_baseline_off_score) / normalizer - np.clip(
        (scores[1] - clipped_baseline_off_score) / normalizer, 0, 1
    )

    return score


def evaluate_model(
    z: torch.Tensor,
    z_pred: torch.Tensor,
    categorical_dimensions: Sequence[int],
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    model_depth: int = 5,
    relative_stopping_threshold: float = 0.005,
    absolute_stopping_threshold: float = 0.001,
    relative_val_stopping_threshold: float = 0.2,
    max_training_epochs: Optional[int] = 100,
    batch_size: int = 128,
    standard_scale: bool = True,
    matching_slack: float = 0.1,
    training_augmentation_gaussian_std: float = 0.005,
    z_mask_values: Optional[Union[torch.Tensor, float, int]] = None,
    verbose: int = 0,
    provided_models=None,
) -> Tuple[
    float,
    float,
    float,
    Tuple[Tuple[float, float], Tuple[float, float]],
    Tuple[Tuple[float, float], Tuple[float, float]],
]:
    """Evaluates a model's identifiability given its predictions and the ground-truth.

    Args:
        z: Tensor containing the ground-truth data.
        z_pred: Tensor containing the predicted data.
        categorical_dimensions: List of ground-truth dimensions containing
            categorical variables.
        train_val_test_split: Tuple containing the relative size of training,
            validation and test split.
        model_depth: Number of layers of the readout model(s).
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
        batch_size: Batch size used during training; this should not be chosen
            to large to avoid getting stuck in local minima.
        standard_scale: Standardize input and output data before fitting the
            regression model(s).
        matching_slack: Slack coefficient for the matching; the larger this
            value is, the more noisy the matching becomes which can help
            avoiding local minima.
        training_augmentation_gaussian_std: Standard deviation of the Gaussian
            noise added to the training data to avoid/reduce overfitting.
        z_mask_values: Values in z that should be masked out. If None, no
            masking is performed.
        verbose: Controls the verbosity level.

    Returns:
        Tuple containing the final overall score, the final score for the continuous
        factors of variation, the final score for the categorical factors of variation,
        and tuples summarizing the readout performance of the model and the ceiling
        performance (i.e., using the ground truth as a model's predictions). Each of the
        last two tuples contains the R² scores and accuracies, where each element is a
        tuple of the form (on-diagonal performance, off-diagonal performance).
    """

    if not z.device == z_pred.device:
        raise RuntimeError("z and z_pred must be on the same device.")

    def _evaluate_model(y, provided_models=provided_models):
        return _internal_evaluate_model(
            z,
            y,
            categorical_dimensions,
            train_val_test_split=train_val_test_split,
            absolute_stopping_threshold=absolute_stopping_threshold,
            relative_stopping_threshold=relative_stopping_threshold,
            relative_val_stopping_threshold=relative_val_stopping_threshold,
            model_depth=model_depth,
            batch_size=batch_size,
            max_training_epochs=max_training_epochs,
            standard_scale=standard_scale,
            matching_slack=matching_slack,
            training_augmentation_gaussian_std=training_augmentation_gaussian_std,
            z_mask_values=z_mask_values,
            verbose=verbose - 1,
            provided_models=provided_models,
        )

    ceiling_r2_scores, ceiling_accuracies, output_models_ceiling = _evaluate_model(
        z, provided_models=None
    )
    r2_scores, accuracies, output_models_normal = _evaluate_model(z_pred)

    continuous_performance = compute_relative_performance(r2_scores, ceiling_r2_scores)
    categorical_performance = compute_relative_performance(
        accuracies, ceiling_accuracies
    )

    performance = 0.0
    if len(categorical_dimensions) > 0:
        performance = performance + categorical_performance * len(
            categorical_dimensions
        )
    if len(categorical_dimensions) < z.shape[-1]:
        performance = continuous_performance * (
            z.shape[-1] - len(categorical_dimensions)
        )
    performance /= z.shape[-1]

    return (
        performance,
        continuous_performance,
        categorical_performance,
        (r2_scores, accuracies),
        (ceiling_r2_scores, ceiling_accuracies),
        output_models_normal,
    )
