import os.path
import time

import torch
import torch.nn.functional as F
import torch.utils.data

import wandb


import src.metrics as metrics
import src.utils.data_utils as data_utils
import src.utils.training_utils as training_utils
from src.utils.wandb_utils import wandb_log, wandb_log_code

from .models import base_models, slot_attention


# TODO: remove this
wandb.login(key="b17ca470c2ce70dd9d6c3ce01c6fc7656633fe91")


def one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    mode,
    epoch,
    reconstruction_term_weight=1,
    consistency_term_weight=1,
    consistency_encoder_term_weight=1,
    consistency_decoder_term_weight=1,
    consistency_scheduler=False,
    consistency_scheduler_step=200,
    use_consistency_loss=True,
    extended_consistency_loss=False,
    detached_latents=False,
    unsupervised_mode=False,
    freq=10,
    **kwargs,
):
    """One epoch of training or testing. Please check main.py for keyword parameters descriptions'."""
    n_samples = len(dataloader.dataset)
    if mode == "train":
        model.train()
    elif mode in ["test_ID", "test_OOD"]:
        model.eval()
    else:
        raise ValueError("mode must be either train or test")

    accum_total_loss = 0
    accum_reconstruction_loss = 0
    accum_slots_loss = 0
    accum_r2_score = 0
    per_latent_r2_score = 0
    accum_consistency_loss = 0
    accum_encoder_consistency_loss = 0
    accum_decoder_consistency_loss = 0
    for batch_idx, (images, true_latents) in enumerate(dataloader):
        total_loss = torch.tensor(0.0, device=device)

        images = images.to(device)
        true_latents = true_latents.to(device)

        if mode == "train":
            optimizer.zero_grad()

        (
            predicted_images,
            predicted_latents,
            predicted_figures,
            sampled_images,
            predicted_z_sampled,
            sampled_figures,
            z_sampled,
            predicted_sampled_images,
        ) = model(
            images,
            use_consistency_loss=use_consistency_loss,
            extended_consistency_loss=extended_consistency_loss,
            detached_latents=detached_latents,
        )

        # calculate slots loss and r2 score for supervised models
        if not unsupervised_mode:
            slots_loss, inds = metrics.hungarian_slots_loss(
                true_latents, predicted_latents, device, reduction="sum"
            )
            accum_slots_loss += slots_loss.item() / n_samples

            avg_r2, raw_r2 = metrics.r2_score(true_latents, predicted_latents, inds)
            accum_r2_score += avg_r2 * len(images) / n_samples
            per_latent_r2_score += raw_r2 * len(images) / n_samples

            # add to total loss
            total_loss += slots_loss

        # calculate reconstruction loss for all models with decoder
        reconstruction_loss = F.mse_loss(predicted_images, images, reduction="sum")
        accum_reconstruction_loss += reconstruction_loss.item() / n_samples

        # add to total loss
        total_loss += reconstruction_loss * reconstruction_term_weight

        # calculate consistency loss
        consistency_encoder_loss, _ = metrics.hungarian_slots_loss(
            z_sampled, predicted_z_sampled, device, reduction="sum"
        )

        accum_encoder_consistency_loss += consistency_encoder_loss.item() / n_samples

        consistency_decoder_loss = F.mse_loss(
            predicted_sampled_images, sampled_images, reduction="sum"
        )
        accum_decoder_consistency_loss += consistency_decoder_loss.item() / n_samples

        consistency_loss = consistency_encoder_loss * consistency_encoder_term_weight
        # add to consistency loss only if extended_consistency_loss is True
        if extended_consistency_loss:
            consistency_loss += (
                consistency_decoder_loss * consistency_decoder_term_weight
            )

        if consistency_scheduler:
            consistency_loss *= min(
                consistency_term_weight, epoch / consistency_scheduler_step
            )
        else:
            consistency_loss *= consistency_term_weight

        accum_consistency_loss += consistency_loss.item() / n_samples

        if use_consistency_loss:
            # add to total loss
            total_loss += consistency_loss

        accum_total_loss += total_loss.item() / n_samples
        if mode == "train":
            total_loss.backward()
            optimizer.step()

    # logging utils
    training_utils.print_metrics_to_console(
        epoch,
        accum_total_loss,
        accum_reconstruction_loss,
        accum_consistency_loss,
        accum_r2_score,
        accum_slots_loss,
        accum_encoder_consistency_loss,
        accum_decoder_consistency_loss,
    )
    if epoch % freq == 0:
        wandb_log(
            **locals(),
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
    lr_scheduler_step,
    reconstruction_term_weight,
    consistency_term_weight,
    consistency_encoder_term_weight,
    consistency_decoder_term_weight,
    consistency_scheduler,
    consistency_scheduler_step,
    use_consistency_loss,
    extended_consistency_loss,
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
    seed,
):
    """
    Run the training and testing. Currently only supports SpritesWorld dataset.
    Check main.py for the description of the parameters.
    """
    passed_args = locals().copy()
    wandb_config = passed_args
    wandb.init(config=wandb_config, project="object_centric_ood")
    wandb.define_metric("test_ID reconstruction loss", summary="min")
    wandb.define_metric("test_OOD reconstruction loss", summary="min")
    wandb_log_code(wandb.run)

    time_created = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    training_utils.set_seed(seed)

    ##### Loading Data #####
    # TODO: Remove all loading from disk
    path = "/mnt/qb/work/bethge/apanfilov27"
    # path = ""
    os.path.isdir(path)
    spriteworld_wrapper = data_utils.SpritesWorldDataWrapper(
        load=True,
        save=False,
        path=path,
    )

    train_loader = spriteworld_wrapper.get_train_loader(**passed_args)
    test_loader_id = spriteworld_wrapper.get_test_id_loader(**passed_args)
    test_loader_ood = spriteworld_wrapper.get_test_ood_loader(**passed_args)

    ##### Loading Identifiability Data #####

    # Why loading data separately? - We experience some issues with generating data on cluster with
    # multiple processes - we don't get separate ground truth figures from gym env. However, we can
    # generate data on a local machine and then load it on cluster.

    identifiability_path = os.path.join(path, "identifiability_data")
    spriteworld_wrapper.identifiability_path = identifiability_path

    identifiability_train_loader = spriteworld_wrapper.get_identifiability_train_loader(
        **passed_args
    )
    identifiability_test_id_loader = (
        spriteworld_wrapper.get_identifiability_test_id_loader(**passed_args)
    )
    identifiability_test_ood_loader = (
        spriteworld_wrapper.get_identifiability_test_ood_loader(**passed_args)
    )
    #######################################

    min_reconstruction_loss_ID = float("inf")
    min_reconstruction_loss_OOD = float("inf")

    in_channels = 3
    if model_name == "SlotMLPAdditive":
        model = base_models.SlotMLPAdditive(in_channels, n_slots, n_slot_latents).to(
            device
        )
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_scheduler_step, gamma=0.5
    )

    for epoch in range(1, epochs + 1):
        total_loss, reconstruction_loss, slots_loss, r2_score = one_epoch(
            model,
            train_loader,
            optimizer,
            mode="train",
            epoch=epoch,
            **passed_args,
        )

        if scheduler.get_last_lr()[0] > 1e-7:
            scheduler.step()

        print("Learning rate: ", optimizer.param_groups[0]["lr"])

        if epoch % 20 == 0:
            if model_name in ["SlotAttention", "SlotMLPAdditive"] and epoch % 500 == 0:
                id_score_id, id_score_ood = metrics.identifiability_score(
                    model,
                    n_slot_latents,
                    identifiability_train_loader,
                    identifiability_test_id_loader,
                    identifiability_test_ood_loader,
                    device,
                )
                wandb.log(
                    {
                        "ID_score_ID": id_score_id,
                        "ID_score_OOD": id_score_ood,
                    },
                    step=epoch,
                )

            (
                id_total_loss,
                id_reconstruction_loss,
                id_slots_loss,
                id_r2_score,
            ) = one_epoch(
                model,
                test_loader_id,
                optimizer,
                mode="test_ID",
                epoch=epoch,
                **passed_args,
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
                mode="test_OOD",
                epoch=epoch,
                **passed_args,
            )

            if id_reconstruction_loss < min_reconstruction_loss_ID:
                min_reconstruction_loss_ID = id_reconstruction_loss
                print(
                    f"\nNew best ID model!\nEpoch: {epoch}\nID reconstruction loss: {id_reconstruction_loss}\n"
                )
                torch.save(
                    model.state_dict(), f"{model_name}_{time_created}_best_id_model.pt"
                )

            if ood_reconstruction_loss < min_reconstruction_loss_OOD:
                min_reconstruction_loss_OOD = ood_reconstruction_loss
                print(
                    f"\nNew best OOD model!\nEpoch: {epoch}\nOOD reconstruction loss: {ood_reconstruction_loss}\n"
                )
                torch.save(
                    model.state_dict(), f"{model_name}_{time_created}_best_ood_model.pt"
                )

    torch.save(model.state_dict(), f"{model_name}_{time_created}_last_train_model.pt")
