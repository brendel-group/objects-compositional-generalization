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
    def __init__(self, path, save, load):
        self.path = path
        self.load = load
        self.save = save

    def get_train_loader(
        self,
        n_samples_train,
        n_samples_truncate,
        n_slots,
        sample_mode_train,
        delta,
        no_overlap,
        batch_size,
        **kwargs,
    ):
        raise NotImplementedError

    def get_test_loader(
        self,
        n_samples_test,
        n_slots,
        sample_mode_test,
        delta,
        no_overlap,
        batch_size,
        **kwargs,
    ):
        raise NotImplementedError


class SpritesWorldDataWrapper(DataWrapper):
    """Wrapper for easy access to train/test loaders for SpritesWorldDataset only."""

    def __init__(self, path, save, load):
        super().__init__(path, save, load)
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
        n_samples_train,
        n_samples_truncate,
        n_slots,
        sample_mode_train,
        delta,
        no_overlap,
        batch_size,
        mixed=False,
        **kwargs,
    ):
        target_path = os.path.join(
            self.path, "train", sample_mode_train, f"{n_slots}_objects"
        )
        if self.load and os.path.exists(target_path) and not mixed:
            train_dataset = PreGeneratedDataset(target_path, n_samples_truncate)
            print(f"Train dataset successfully loaded from {target_path}.")
        elif mixed:
            # go on directory back
            target_path = os.path.join(self.path, "train", sample_mode_train, "mixed")
            train_dataset = MixedDataset(target_path)
            print(f"Train dataset successfully loaded from {target_path}.")
        else:
            train_dataset = data.SpriteWorldDataset(
                n_samples_train,
                n_slots,
                self.config,
                sample_mode=sample_mode_train,
                delta=delta,
                no_overlap=no_overlap,
                transform=transforms.Compose(
                    [transforms.ToPILImage(), transforms.ToTensor()]
                ),
            )
            if self.save:
                dump_generated_dataset(train_dataset, target_path)
                print(f"Train dataset successfully saved to {target_path}.")

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
        n_samples_test,
        n_slots,
        sample_mode_test,
        delta,
        no_overlap,
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
        elif mixed:
            # go on directory back
            target_path = os.path.join(self.path, "test", sample_mode_test, "mixed")
            print(
                f"Test {sample_mode_test} dataset successfully loaded from {target_path}."
            )
            test_dataset = MixedDataset(target_path)
        else:
            test_dataset = data.SpriteWorldDataset(
                n_samples_test,
                n_slots,
                self.config,
                sample_mode=sample_mode_test,
                delta=delta,
                no_overlap=no_overlap,
                transform=transforms.Compose(
                    [transforms.ToPILImage(), transforms.ToTensor()]
                ),
            )
            if self.save:
                dump_generated_dataset(test_dataset, target_path)
                print(f"Test dataset successfully saved to {self.path}.")

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn_normalizer(
                b, self.min_offset, self.scale, mixed=mixed
            ),
        )

        return test_loader


class KubricWrapper(DataWrapper):
    def __init__(self, path, save, load):
        super().__init__(path, save, load)

        self.config = configs.KubricConfig()
        self.__min_offset = None
        self.__scale = None

    @property
    def scale(self):
        if self.__scale is None:
            scale = torch.FloatTensor(
                [rng.max - rng.min for rng in self.config.get_ranges().values()]
            ).reshape(1, 1, -1)
            # excluding z latent
            scale = torch.cat([scale[:, :, :2], scale[:, :, 3:]], dim=-1)
            self.__scale = scale
        return self.__scale

    @property
    def min_offset(self):
        if self.__min_offset is None:
            min_offset = torch.FloatTensor(
                [rng.min for rng in self.config.get_ranges().values()]
            ).reshape(1, 1, -1)
            # excluding z latent
            min_offset = torch.cat([min_offset[:, :, :2], min_offset[:, :, 3:]], dim=-1)
            self.__min_offset = min_offset
        return self.__min_offset

    def get_train_loader(
        self,
        n_samples_train,
        n_slots,
        sample_mode_train,
        n_samples_truncate,
        delta,
        no_overlap,
        batch_size,
        **kwargs,
    ):
        if self.load and os.path.exists(
            os.path.join(self.path, "train", sample_mode_train)
        ):
            train_dataset = PreGeneratedDataset(
                os.path.join(self.path, "train", sample_mode_train), n_samples_truncate
            )
            train_dataset.latents = torch.cat(
                [train_dataset.latents[:, :, :2], train_dataset.latents[:, :, 3:]],
                dim=-1,
            )
            print(
                f"Train dataset successfully loaded from {os.path.join(self.path, 'train', sample_mode_train)}."
            )
        else:
            kubric_path = "SOME_PATH"

            train_dataset = data.KubricImagesDataset(
                os.path.join(kubric_path, "train", sample_mode_train)
            )
            if self.save:
                dump_generated_dataset(
                    train_dataset, os.path.join(self.path, "train", sample_mode_train)
                )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn_normalizer(b, self.min_offset, self.scale),
        )

        return train_loader

    def get_test_loader(
        self,
        n_samples_test,
        n_slots,
        sample_mode_test,
        delta,
        no_overlap,
        batch_size,
        **kwargs,
    ):
        if self.load and os.path.exists(
            os.path.join(self.path, "test", sample_mode_test)
        ):
            test_dataset = PreGeneratedDataset(
                os.path.join(self.path, "test", sample_mode_test)
            )
            test_dataset.latents = torch.cat(
                [test_dataset.latents[:, :, :2], test_dataset.latents[:, :, 3:]], dim=-1
            )
            print(
                f"Test dataset successfully loaded from {os.path.join(self.path, 'test', sample_mode_test)}."
            )
        else:
            kubric_path = "SOME_PATH"

            test_dataset = data.KubricImagesDataset(
                os.path.join(kubric_path, "test", sample_mode_test)
            )
            if self.save:
                dump_generated_dataset(
                    test_dataset, os.path.join(self.path, "test", sample_mode_test)
                )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn_normalizer(b, self.min_offset, self.scale),
        )

        return test_loader


def get_wrapper(dataset_name, path, save=False, load=False):
    if dataset_name == "dsprites":
        return SpritesWorldDataWrapper(path, save, load)
    elif dataset_name == "kubric":
        return KubricWrapper(path, save, load)
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")
