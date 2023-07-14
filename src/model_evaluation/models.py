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

from typing import Callable, List, Optional

import torch
from torch import nn


class Model(nn.Module):
    """Model that infers both continuous and categorical variables.

    Args:
        n_slots: Number of slots.
        pred_slot_dimensionality: Dimensionality of the predicted slot.
        gt_continuous_slot_dimensionality: Dimensionality of the continuous variables
            in the ground truth, i.e., how many continuous variables to infer.
        continuous_slot_readout_fn: Function that takes the predicted slot
            dimensionality and the ground-truth slot dimensionality and returns
            a readout module.
        gt_num_categories: Number of categories for each categorical variable.
    """

    def __init__(
        self,
        n_slots: int,
        pred_slot_dimensionality: int,
        gt_continuous_slot_dimensionality: Optional[int],
        continuous_slot_readout_fn: Callable[[int, int], nn.Module],
        gt_num_categories: Optional[List[int]],
        categorical_slot_readout_fn: Callable[[int, int], nn.Module],
        share_readout: bool = False,
    ):
        super().__init__()
        if gt_continuous_slot_dimensionality is None:
            self._continuous_model = lambda *args, **kwargs: None
        else:
            self._continuous_model = ContinuousModel(
                n_slots,
                pred_slot_dimensionality,
                gt_continuous_slot_dimensionality,
                continuous_slot_readout_fn,
                share_readout,
            )

        if gt_num_categories is None:
            self._categorical_model = lambda *args, **kwargs: None
        else:
            self._categorical_model = CategoricalModel(
                n_slots,
                pred_slot_dimensionality,
                gt_num_categories,
                categorical_slot_readout_fn,
                share_readout,
            )

    def forward(self, x: torch.Tensor):
        y = self._continuous_model(x)
        ps = self._categorical_model(x)

        return y, ps


class ContinuousModel(nn.Module):
    """Readout module for continuous variables.

    Args:
        n_slots: Number of slots.
        pred_slot_dimensionality: Dimensionality of the predicted slot.
        gt_slot_dimensionality: Dimensionality of the continuous variables
            in the ground truth, i.e., how many continuous variables to infer.
        slot_readout_fn: Function that takes the predicted slot dimensionality
            and the ground-truth slot dimensionality and returns a readout
            module.
        share_readout: Whether to share the readout module across all slots.
    """

    def __init__(
        self,
        n_slots: int,
        pred_slot_dimensionality: int,
        gt_slot_dimensionality: int,
        slot_readout_fn: Callable[[int, int], nn.Module],
        share_readout: bool = False,
    ):
        super().__init__()
        self._share_readout = share_readout
        if self._share_readout:
            self._readout = slot_readout_fn(
                pred_slot_dimensionality, gt_slot_dimensionality
            )
        else:
            self._readouts = nn.ModuleList(
                [
                    slot_readout_fn(pred_slot_dimensionality, gt_slot_dimensionality)
                    for _ in range(n_slots)
                ]
            )

    def forward(self, x: torch.Tensor):
        if self._share_readout:
            bs, ns = x.shape[:2]
            x = x.view((bs * ns, *x.shape[2:]))
            y = self._readout(x)
            y = y.view(bs, ns, *y.shape[1:])
        else:
            if x.shape[-2] != len(self._readouts):
                raise ValueError("Invalid number of slots.")
            y = []
            for i in range(x.shape[-2]):
                y.append(
                    torch.stack(
                        [self._readouts[i](x[:, j]) for j in range(x.shape[-2])], -2
                    )
                )
            y = torch.stack(y, -3)
        return y


class Parallel(nn.Module):
    """Module that applies a list of modules in parallel."""

    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        return [m(*args, **kwargs) for m in self.models]


class CategoricalModel(nn.Module):
    """Readout module for categorical variables.

    Args:
        n_slots: Number of slots.
        pred_slot_dimensionality: Dimensionality of the predicted slot.
        gt_num_categories: Number of categories for each ground-truth variable.
        slot_readout_fn: Function that takes the predicted slot dimensionality
            and the number of categories and returns a readout module.
        share_readout: Whether to share the readout module across all slots.
    """

    def __init__(
        self,
        n_slots: int,
        pred_slot_dimensionality: int,
        gt_num_categories: List[int],
        slot_readout_fn: Callable[[int, int], nn.Module],
        share_readout: bool = False,
    ):
        super().__init__()
        self._share_readout = share_readout
        if self._share_readout:
            self._readout = Parallel(
                *[
                    slot_readout_fn(pred_slot_dimensionality, nc)
                    for nc in gt_num_categories
                ]
            )
        else:
            self._readouts = nn.ModuleList(
                [
                    Parallel(
                        *[
                            slot_readout_fn(pred_slot_dimensionality, nc)
                            for nc in gt_num_categories
                        ]
                    )
                    for _ in range(n_slots)
                ]
            )

    def forward(self, x: torch.Tensor):
        if self._share_readout:
            bs, ns = x.shape[:2]
            x = x.view((bs * ns, *x.shape[2:]))
            ps = self._readout(x)
            ps = [p.view(bs, ns, *p.shape[1:]) for p in ps]
        else:
            if x.shape[-2] != len(self._readouts):
                raise ValueError("Invalid number of slots.")
            raise NotImplementedError("Not yet implemented.")
        return ps
