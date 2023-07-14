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

from typing import Callable, Sequence, Tuple

import torch


class TransformableTensorDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, ...]]):
    """Extension of torch.utils.data.TensorDataset allowing transformations.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        tensors (Tensor): tensors that have the same size of the first dimension.
        transform: A function/transform that takes in a tuple of tensors and returns a
            tuple of transformed tensors.
    """

    tensors: Sequence[torch.Tensor]

    def __init__(
        self,
        tensors: Sequence[torch.Tensor],
        transform: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    ) -> None:
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(tuple(tensor[index] for tensor in self.tensors))

    def __len__(self):
        return self.tensors[0].size(0)
