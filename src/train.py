import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import wandb

wandb.login(key="b17ca470c2ce70dd9d6c3ce01c6fc7656633fe91")

from . import config
from . import data
from .models import base_models
from .training_utils import (
    calculate_r2_score,
    matched_slots_loss,
    collate_fn_normalizer,
)

from .wandb_utils import wandb_log


def one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    mode="train",
    epoch=0,
    reduction="sum",
    use_sampled_loss=True,
    detached_latents=False,
    unsupervised_mode=False,
    freq=10,
):
    n_samples = len(dataloader.dataset)
    if mode == "train":
        model.train()
    elif mode in ["test_ID", "test_OOD"]:
        model.eval()
    else:
        raise ValueError("mode must be either train or test")

    accum_total_loss = 0
    accum_slots_loss = 0
    accum_sampled_loss = 0
    accum_reconstruction_loss = 0
    r2_score = 0
    per_latent_r2_score = 0
    for batch_idx, (data, true_latents) in enumerate(dataloader):
        reconstruction_loss = 0
        slots_loss = 0
        sampled_loss = 0
        predicted_figures = None
        predicted_images = None
        sampled_images = None
        sampled_figures = None

        data = data.to(device)
        true_latents = true_latents.to(device)

        if mode == "train":
            optimizer.zero_grad()

        if model.model_name == "SlotMLPMonolithic":
            predicted_images, predicted_latents = model(data)
        elif model.model_name == "SlotMLPAdditive":
            if mode == "train" and use_sampled_loss:
                (
                    predicted_images,
                    predicted_latents,
                    predicted_figures,
                    z_hat,
                    sampled_images,
                    sampled_figures,
                    sampled_z,
                ) = model(
                    data,
                    use_sampled_loss=use_sampled_loss,
                    detached_latents=(detached_latents and not unsupervised_mode),
                )
            else:
                predicted_images, predicted_latents, predicted_figures = model(data)
        elif model.model_name == "SlotMLPEncoder":
            predicted_latents = model(data)
        elif model.model_name == "SlotMLPAdditiveDecoder":
            predicted_images, predicted_figures = model(true_latents)
        elif model.model_name == "SlotMLPMonolithicDecoder":
            predicted_images = model(true_latents)

        if (
            model.model_name
            not in [
                "SlotMLPAdditiveDecoder",
                "SlotMLPMonolithicDecoder",
            ]
            and not unsupervised_mode
        ):
            slots_loss, inds = matched_slots_loss(
                predicted_latents, true_latents, device, reduction=reduction
            )
            accum_slots_loss += slots_loss.item() / n_samples

            avg_r2, raw_r2 = calculate_r2_score(true_latents, predicted_latents, inds)
            r2_score += avg_r2 * len(data) / n_samples
            per_latent_r2_score += raw_r2 * len(data) / n_samples

        if model.model_name != "SlotMLPEncoder":
            reconstruction_loss = F.mse_loss(
                predicted_images, data, reduction=reduction
            )
            accum_reconstruction_loss += reconstruction_loss.item() / n_samples

        if (
            model.model_name in ["SlotMLPAdditive"]
            and mode == "train"
            and use_sampled_loss
        ):
            sampled_loss, _ = matched_slots_loss(
                z_hat, sampled_z, device, reduction=reduction
            )
            accum_sampled_loss += sampled_loss.item() / n_samples

        total_loss = (
            reconstruction_loss * 0.01
            + slots_loss
            + sampled_loss * max(1, epoch / 4000)
        )

        accum_total_loss += total_loss.item() / n_samples

        if mode == "train":
            total_loss.backward()
            optimizer.step()

    print(
        "====> Epoch: {} Average loss: {:.4f}, r2 score {:.4f}".format(
            epoch,
            accum_total_loss,
            r2_score,
        )
    )
    if epoch % freq == 0:
        wandb_log(
            mode,
            epoch,
            freq,
            total_loss=accum_total_loss,
            slots_loss=accum_slots_loss,
            reconstruction_loss=accum_reconstruction_loss,
            r2_score=r2_score,
            r2_score_raw=per_latent_r2_score,
            true_images=data,
            predicted_images=predicted_images,
            predicted_figures=predicted_figures,
            sampled_loss=accum_sampled_loss,
            sampled_images=sampled_images,
            sampled_figures=sampled_figures,
        )

    return (
        accum_total_loss,
        accum_reconstruction_loss,
        accum_slots_loss,
        r2_score,
    )


def run(
    *,
    model_name,
    device,
    epochs,
    batch_size,
    lr,
    weight_decay,
    reduction,
    use_sampled_loss,
    unsupervised_mode,
    detached_latents,
    n_samples_train,
    n_samples_test,
    n_slots,
    n_slot_latents,
    no_overlap,
    sample_mode_train,
    sample_mode_test_id,
    sample_mode_test_ood,
    delta,
    in_channels,
    seed,
):
    """
    Run the training and testing. Currently only supports SpritesWorld dataset.

    Args:
        model_name: Model to use. One of the models defined in base_models.py.
        device: Device to use. Either "cpu" or "cuda".
        epochs: Number of epochs to train for.
        batch_size: Batch size to use.
        lr: Learning rate to use.
        weight_decay: Weight decay to use.
        reduction: Reduction to use for loss. Either "sum" or "mean".
        use_sampled_loss: Whether to use sampled loss.
        unsupervised_mode: Turns model to Autoencoder mode (no slots loss).
        detached_latents: Detach latents from encoder or not.
        n_samples_train: Number of samples in training dataset.
        n_samples_test: Number of samples in testing dataset (ID and OOD).
        n_slots: Number of slots, i.e. objects in scene.
        n_slot_latents: Number of latents per slot. Right now, this is fixed to 8.
        no_overlap: Whether to allow overlapping figures.
        sample_mode_train: Sampling mode for training dataset.
        sample_mode_test_id: Sampling mode for ID testing dataset.
        sample_mode_test_ood: Sampling mode for OOD testing dataset.
        delta: Delta for "diagonal" and "off_diagonal" dataset.
        in_channels: Number of channels in input image.
        seed: Random seed to use.
    """
    wandb_config = {
        "model_name": model_name,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "reduction": reduction,
        "use_sampled_loss": use_sampled_loss,
        "unsupervised_mode": unsupervised_mode,
        "detached_latents": detached_latents,
        "n_samples_train": n_samples_train,
        "n_samples_test": n_samples_test,
        "n_slots": n_slots,
        "n_slot_latents": n_slot_latents,
        "no_overlap": no_overlap,
        "sample_mode_train": sample_mode_train,
        "sample_mode_test_id": sample_mode_test_id,
        "sample_mode_test_ood": sample_mode_test_ood,
        "delta": delta,
        "in_channels": in_channels,
        "seed": seed,
    }
    wandb.init(config=wandb_config, project="object_centric_ood")

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    cfg = config.SpriteWorldConfig()
    min_offset = torch.FloatTensor(
        [rng.min for rng in cfg.get_ranges().values()]
    ).reshape(1, 1, -1)
    scale = torch.FloatTensor(
        [rng.max - rng.min for rng in cfg.get_ranges().values()]
    ).reshape(1, 1, -1)
    scale[scale == 0] = 1

    if model_name == "SlotMLPAdditive":
        model = base_models.SlotMLPAdditive(in_channels, n_slots, n_slot_latents).to(
            device
        )
    elif model_name == "SlotMLPMonolithic":
        model = base_models.SlotMLPMonolithic(in_channels, n_slots, n_slot_latents).to(
            device
        )
    elif model_name == "SlotMLPEncoder":
        model = base_models.SlotMLPEncoder(in_channels, n_slots, n_slot_latents).to(
            device
        )
    elif model_name == "SlotMLPAdditiveDecoder":
        model = base_models.SlotMLPAdditiveDecoder(
            in_channels, n_slots, n_slot_latents
        ).to(device)
    elif model_name == "SlotMLPMonolithicDecoder":
        model = base_models.SlotMLPMonolithicDecoder(
            in_channels, n_slots, n_slot_latents
        ).to(device)
    wandb.watch(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.8,
        verbose=True,
        cooldown=2,
        threshold=1e-5,
    )

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = data.SpriteWorldDataset(
        n_samples_train,
        n_slots,
        cfg,
        sample_mode=sample_mode_train,
        delta=delta,
        no_overlap=no_overlap,
        transform=transform,
    )
    test_dataset_id = data.SpriteWorldDataset(
        n_samples_test,
        n_slots,
        cfg,
        sample_mode=sample_mode_test_id,
        delta=delta,
        no_overlap=no_overlap,
        transform=transform,
    )
    test_dataset_ood = data.SpriteWorldDataset(
        n_samples_test,
        n_slots,
        cfg,
        sample_mode=sample_mode_test_ood,
        delta=delta,
        no_overlap=no_overlap,
        transform=transform,
    )

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn_normalizer(b, min_offset, scale),
    )
    test_loader_id = torch.utils.data.DataLoader(
        test_dataset_id,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_normalizer(b, min_offset, scale),
    )
    test_loader_ood = torch.utils.data.DataLoader(
        test_dataset_ood,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_normalizer(b, min_offset, scale),
    )

    min_reconstruction_loss_ID = float("inf")
    min_reconstruction_loss_OOD = float("inf")
    flag = False
    for epoch in range(1, epochs + 1):
        total_loss, reconstruction_loss, slots_loss, r2_score = one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mode="train",
            epoch=epoch,
            reduction=reduction,
            use_sampled_loss=use_sampled_loss,
            unsupervised_mode=unsupervised_mode,
            detached_latents=detached_latents,
        )

        if epoch % 20 == 0:
            (
                id_total_loss,
                id_reconstruction_loss,
                id_slots_loss,
                id_r2_score,
            ) = one_epoch(
                model,
                test_loader_id,
                optimizer,
                device,
                mode="test_ID",
                epoch=epoch,
                unsupervised_mode=unsupervised_mode,
                reduction=reduction,
            )
            (
                ood_total_loss,
                ood_reconstruction_loss,
                ood_slots_loss,
                ood_r2_score,
            ) = one_epoch(
                model,
                test_loader_ood,
                optimizer,
                device,
                mode="test_OOD",
                epoch=epoch,
                unsupervised_mode=unsupervised_mode,
                reduction=reduction,
            )
            print("OOD slots loss: ", ood_slots_loss)
            if ood_slots_loss < 0.15 or ood_reconstruction_loss < 15 or flag:
                scheduler.step(ood_total_loss)
                flag = True
                print("Learning rate: ", optimizer.param_groups[0]["lr"])
            scheduler.step(total_loss)
            print("Learning rate: ", optimizer.param_groups[0]["lr"])

            if id_reconstruction_loss < min_reconstruction_loss_ID:
                min_reconstruction_loss_ID = id_reconstruction_loss
                print()
                print("New best ID model!")
                print("Epoch:", epoch)
                print("ID reconstruction loss:", id_reconstruction_loss)
                print()
                torch.save(model.state_dict(), f"{model_name}_best_id_model.pt")

            if ood_reconstruction_loss < min_reconstruction_loss_OOD:
                min_reconstruction_loss_OOD = ood_reconstruction_loss
                print()
                print("New best OOD model!")
                print("Epoch:", epoch)
                print("OOD reconstruction loss:", ood_reconstruction_loss)
                print()
                torch.save(model.state_dict(), f"{model_name}_best_ood_model.pt")

    torch.save(model.state_dict(), f"{model_name}_last_train_model.pt")
