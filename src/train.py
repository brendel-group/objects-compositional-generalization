import os.path
import time

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import wandb

wandb.login(key="b17ca470c2ce70dd9d6c3ce01c6fc7656633fe91")

from . import config
from . import data
from .models import base_models
from .models import slot_attention

from src.utils.training_utils import (
    collate_fn_normalizer,
    set_seed,
)
from .metrics import r2_score, hungarian_slots_loss

from src.utils.wandb_utils import wandb_log

from src.utils.data_utlis import dump_generated_dataset, PreGeneratedDataset


def one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    mode="train",
    epoch=0,
    reduction="sum",
    reconstruction_term_weight=1,
    consistency_term_weight=1,
    use_consistency_loss=True,
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
    accum_consistency_loss = 0
    accum_reconstruction_loss = 0
    accum_r2_score = 0
    per_latent_r2_score = 0
    for batch_idx, (data, true_latents) in enumerate(dataloader):
        reconstruction_loss = 0
        slots_loss = 0
        consistency_loss = 0
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
        elif model.model_name in ["SlotMLPAdditive", "SlotAttention"]:
            if mode == "train" and use_consistency_loss:
                (
                    predicted_images,
                    predicted_latents,
                    predicted_figures,
                    sampled_images,
                    predicted_z_sampled,
                    sampled_figures,
                    z_sampled,
                ) = model(
                    data,
                    use_consistency_loss=use_consistency_loss,
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
                "SlotAttention",
            ]
            and not unsupervised_mode
        ):
            slots_loss, inds = hungarian_slots_loss(
                predicted_latents, true_latents, device, reduction=reduction
            )
            accum_slots_loss += slots_loss.item() / n_samples

            avg_r2, raw_r2 = r2_score(true_latents, predicted_latents, inds)
            accum_r2_score += avg_r2 * len(data) / n_samples
            per_latent_r2_score += raw_r2 * len(data) / n_samples

        if model.model_name != "SlotMLPEncoder":
            reconstruction_loss = F.mse_loss(
                predicted_images, data, reduction=reduction
            )
            accum_reconstruction_loss += reconstruction_loss.item() / n_samples

        if (
            model.model_name in ["SlotMLPAdditive", "SlotAttention"]
            and mode == "train"
            and use_consistency_loss
        ):
            consistency_loss, _ = hungarian_slots_loss(
                z_sampled, predicted_z_sampled, device, reduction=reduction
            )
            accum_consistency_loss += consistency_loss.item() / n_samples

        total_loss = (
            reconstruction_loss * reconstruction_term_weight
            + slots_loss
            + consistency_loss * consistency_term_weight
        )

        accum_total_loss += total_loss.item() / n_samples

        if mode == "train":
            total_loss.backward()
            optimizer.step()

    print(
        "====> Epoch: {} Average loss: {:.4f}, r2 score {:.4f} \n "
        "       reconstruction loss {:.4f} consistency loss {:.4f} slots loss {:.4f}".format(
            epoch,
            accum_total_loss,
            accum_r2_score,
            accum_reconstruction_loss,
            accum_consistency_loss,
            accum_slots_loss,
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
            r2_score=accum_r2_score,
            r2_score_raw=per_latent_r2_score,
            true_images=data,
            predicted_images=predicted_images,
            predicted_figures=predicted_figures,
            consistency_loss=accum_consistency_loss,
            sampled_images=sampled_images,
            sampled_figures=sampled_figures,
        )

    return (
        accum_total_loss,
        accum_reconstruction_loss,
        accum_slots_loss,
        accum_r2_score,
    )


def run(
    *,
    model_name,
    device,
    epochs,
    batch_size,
    lr,
    weight_decay,
    lr_scheduler_step,
    reduction,
    reconstruction_term_weight,
    consistency_term_weight,
    use_consistency_loss,
    warmup,
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
        lr_scheduler_step: How often to decrease learning rate.
        reduction: Reduction to use for loss. Either "sum" or "mean".
        reconstruction_term_weight: Weight for reconstruction term in total loss.
        consistency_term_weight: Weight for consistency term in total loss.
        use_consistency_loss: Whether to use consistency loss.
        warmup: Whether to use warmup.
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
        "lr_scheduler_step": lr_scheduler_step,
        "reduction": reduction,
        "reconstruction_term_weight": reconstruction_term_weight,
        "consistency_term_weight": consistency_term_weight,
        "use_consistency_loss": use_consistency_loss,
        "warmup": warmup,
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
    time_created = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    set_seed(seed)

    cfg = config.SpriteWorldConfig()
    min_offset = torch.FloatTensor(
        [rng.min for rng in cfg.get_ranges().values()]
    ).reshape(1, 1, -1)
    scale = torch.FloatTensor(
        [rng.max - rng.min for rng in cfg.get_ranges().values()]
    ).reshape(1, 1, -1)

    # excluding fixed latents (rotation and two colour channels
    min_offset = torch.cat([min_offset[:, :, :-4], min_offset[:, :, -3:-2]], dim=-1)
    scale = torch.cat([scale[:, :, :-4], scale[:, :, -3:-2]], dim=-1)

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
    elif model_name == "SlotAttention":
        encoder = slot_attention.SlotAttentionEncoder(
            resolution=(64, 64), hid_dim=n_slot_latents
        ).to(device)
        decoder = slot_attention.SlotAttentionDecoder(
            hid_dim=n_slot_latents, resolution=(64, 64)
        ).to(device)
        model = slot_attention.SlotAttentionAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            num_slots=n_slots,
            num_iterations=3,
            hid_dim=n_slot_latents,
        ).to(device)

    wandb.watch(model)

    transform = transforms.Compose([transforms.ToTensor()])

    # TODO: Remove all loading from disk
    path = "/mnt/qb/work/bethge/apanfilov27"
    # path = ""
    load = True
    save = False
    os.path.isdir(path)

    if load and os.path.isdir(os.path.join(path, "train")):
        train_dataset = PreGeneratedDataset(os.path.join(path, "train"))
        print("Train dataset successfully loaded from disk.")
    else:
        train_dataset = data.SpriteWorldDataset(
            n_samples_train,
            n_slots,
            cfg,
            sample_mode=sample_mode_train,
            delta=delta,
            no_overlap=no_overlap,
            transform=transform,
        )
        if save:
            dump_generated_dataset(train_dataset, os.path.join(path, "train"))

    if load and os.path.isdir(os.path.join(path, "test_id")):
        test_dataset_id = PreGeneratedDataset(os.path.join(path, "test_id"))
        print("Test ID dataset successfully loaded from disk.")
    else:
        test_dataset_id = data.SpriteWorldDataset(
            n_samples_test,
            n_slots,
            cfg,
            sample_mode=sample_mode_test_id,
            delta=delta,
            no_overlap=no_overlap,
            transform=transform,
        )
        if save:
            dump_generated_dataset(test_dataset_id, os.path.join(path, "test_id"))

    if load and os.path.isdir(os.path.join(path, "test_ood")):
        test_dataset_ood = PreGeneratedDataset(os.path.join(path, "test_ood"))
        print("Test OOD dataset successfully loaded from disk.")
    else:
        test_dataset_ood = data.SpriteWorldDataset(
            n_samples_test,
            n_slots,
            cfg,
            sample_mode=sample_mode_test_ood,
            delta=delta,
            no_overlap=no_overlap,
            transform=transform,
        )
        if save:
            dump_generated_dataset(test_dataset_ood, os.path.join(path, "test_ood"))

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

    # lr=5e-6 for warming up
    if warmup:
        init_lr = 5e-6
    else:
        init_lr = lr

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    if warmup:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_scheduler_step, gamma=0.5
        )

    for epoch in range(1, epochs + 1):
        total_loss, reconstruction_loss, slots_loss, r2_score = one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mode="train",
            epoch=epoch,
            reduction=reduction,
            reconstruction_term_weight=reconstruction_term_weight,
            consistency_term_weight=consistency_term_weight,
            use_consistency_loss=use_consistency_loss,
            unsupervised_mode=unsupervised_mode,
            detached_latents=detached_latents,
        )

        if warmup and scheduler.get_last_lr()[0] > lr:
            optimizer.param_groups[0]["lr"] = lr
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_scheduler_step, gamma=0.5
            )
            warmup = False

        if scheduler.get_last_lr()[0] > 1e-7:
            scheduler.step()

        print("Learning rate: ", optimizer.param_groups[0]["lr"])

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
                reduction=reduction,
                reconstruction_term_weight=reconstruction_term_weight,
                consistency_term_weight=consistency_term_weight,
                unsupervised_mode=unsupervised_mode,
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
                reduction=reduction,
                reconstruction_term_weight=reconstruction_term_weight,
                consistency_term_weight=consistency_term_weight,
                unsupervised_mode=unsupervised_mode,
            )

            if id_reconstruction_loss < min_reconstruction_loss_ID:
                min_reconstruction_loss_ID = id_reconstruction_loss
                print()
                print("New best ID model!")
                print("Epoch:", epoch)
                print("ID reconstruction loss:", id_reconstruction_loss)
                print()
                torch.save(
                    model.state_dict(), f"{model_name}_{time_created}_best_id_model.pt"
                )

            if ood_reconstruction_loss < min_reconstruction_loss_OOD:
                min_reconstruction_loss_OOD = ood_reconstruction_loss
                print()
                print("New best OOD model!")
                print("Epoch:", epoch)
                print("OOD reconstruction loss:", ood_reconstruction_loss)
                print()
                torch.save(
                    model.state_dict(), f"{model_name}_{time_created}_best_ood_model.pt"
                )

    torch.save(model.state_dict(), f"{model_name}_{time_created}_last_train_model.pt")
