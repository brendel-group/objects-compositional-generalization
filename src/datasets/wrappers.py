import os

import torch
from src.datasets import configs, data
from src.datasets.utils import (
    MixedDataset,
    PreGeneratedDataset,
    collate_fn_normalizer,
    dump_generated_dataset,
)
from torchvision import transforms as transforms


class DataWrapper:
    def __init__(self, path):
        self.path = path

    def get_train_loader(
        self,
        n_slots,
        sample_mode_train,
        batch_size,
        **kwargs,
    ):
        raise NotImplementedError

    def get_test_loader(
        self,
        n_slots,
        sample_mode_test,
        batch_size,
        **kwargs,
    ):
        raise NotImplementedError


class SpritesWorldDataWrapper(DataWrapper):
    """Wrapper for easy access to train/test loaders for SpritesWorldDataset only."""

    def __init__(self, path):
        super().__init__(path)
        self.config = configs.SpriteWorldConfig()
        self.__scale = None
        self.__min_offset = None

    @property
    def scale(self):
        if self.__scale is None:
            scale = torch.FloatTensor(
                [rng.max - rng.min for rng in self.config.get_ranges().values()]
            ).reshape(1, 1, -1)
            # excluding fixed latents (rotation and two colour channels)
            scale = torch.cat([scale[:, :, :-4], scale[:, :, -3:-2]], dim=-1)
            self.__scale = scale

        return self.__scale

    @property
    def min_offset(self):
        if self.__min_offset is None:
            min_offset = torch.FloatTensor(
                [rng.min for rng in self.config.get_ranges().values()]
            ).reshape(1, 1, -1)
            # excluding fixed latents (rotation and two colour channels)
            min_offset = torch.cat(
                [min_offset[:, :, :-4], min_offset[:, :, -3:-2]], dim=-1
            )
            self.__min_offset = min_offset

        return self.__min_offset

    def get_train_loader(
        self,
        n_slots,
        sample_mode_train,
        batch_size,
        mixed=False,
        **kwargs,
    ):
        target_path = os.path.join(
            self.path, "train", sample_mode_train, f"{n_slots}_objects"
        )
        if os.path.exists(target_path) and not mixed:
            train_dataset = PreGeneratedDataset(target_path)
            print(f"Train dataset successfully loaded from {target_path}.")
        elif mixed and os.path.exists(
            os.path.join(self.path, "train", sample_mode_train, "mixed")
        ):
            target_path = os.path.join(self.path, "train", sample_mode_train, "mixed")
            train_dataset = MixedDataset(target_path)
            print(f"Train dataset successfully loaded from {target_path}.")
        else:
            raise ValueError(
                f"Train dataset for {sample_mode_train} and {n_slots} objects not found."
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn_normalizer(
                b, self.min_offset, self.scale, mixed=mixed
            ),
        )

        return train_loader

    def get_test_loader(
        self,
        n_slots,
        sample_mode_test,
        batch_size,
        mixed=False,
        **kwargs,
    ):
        target_path = os.path.join(
            self.path, "test", sample_mode_test, f"{n_slots}_objects"
        )
        if self.load and os.path.exists(target_path) and not mixed:
            test_dataset = PreGeneratedDataset(target_path)
            print(
                f"Test {sample_mode_test} dataset successfully loaded from {target_path}."
            )
        elif mixed and os.path.exists(
            os.path.join(self.path, "test", sample_mode_test, "mixed")
        ):
            # go on directory back
            target_path = os.path.join(self.path, "test", sample_mode_test, "mixed")
            print(
                f"Test {sample_mode_test} dataset successfully loaded from {target_path}."
            )
            test_dataset = MixedDataset(target_path)
        else:
            raise ValueError(
                f"Test dataset for {sample_mode_test} and {n_slots} objects not found."
            )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn_normalizer(
                b, self.min_offset, self.scale, mixed=mixed
            ),
        )

        return test_loader


def get_wrapper(dataset_name, path, save=False, load=False):
    if dataset_name == "dsprites":
        return SpritesWorldDataWrapper(path, save, load)
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")
