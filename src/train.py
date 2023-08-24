import os.path
import time

import torch
import torch.utils.data

# import wandb

import src.datasets.wrappers
import src.metrics as metrics
import src.models
import src.datasets.utils as data_utils
import src.utils.training_utils as training_utils
import src.utils.wandb_utils as wandb_utils

from contextlib import nullcontext

from .models import base_models, slot_attention

# # TODO: remove this
# wandb.login(key="b17ca470c2ce70dd9d6c3ce01c6fc7656633fe91")


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
    consistency_ignite_epoch=0,
    use_consistency_loss=True,
    extended_consistency_loss=False,
    unsupervised_mode=False,
    freq=1,
    **kwargs,
):
    """One epoch of training or testing. Please check main.py for keyword parameters descriptions'."""
    print(f"Number of samples: {len(dataloader.dataset)}")
    if mode == "train":
        model.train()
    elif mode in ["test_ID", "test_OOD", "test_RDM"]:
        model.eval()
    else:
        raise ValueError("mode must be either train or test")

    accum_total_loss = 0
    accum_model_loss = 0
    accum_reconstruction_loss = 0
    accum_reconstruction_r2 = 0
    accum_slots_loss = 0
    accum_r2_score = 0
    accum_ari_score = 0
    per_latent_r2_score = 0
    accum_consistency_loss = 0
    accum_consistency_encoder_loss = 0
    accum_consistency_decoder_loss = 0
    for batch_idx, (images, true_latents) in enumerate(dataloader):
        total_loss = torch.tensor(0.0, device=device)
        # adjust overall epoch loss to match mean over all samples
        accum_adjustment = len(images) / len(dataloader.dataset)

        # first dimensions contain separate objects, last dimension is the final image ("sum" of objects)
        images = images.to(device)
        true_figures = images[:, :-1, ...]
        images = images[:, -1, ...].squeeze(1)
        true_latents = true_latents.to(device)

        if mode == "train":
            optimizer.zero_grad()

        output_dict = model(
            images,
            use_consistency_loss=use_consistency_loss
            * (epoch >= consistency_ignite_epoch),
            extended_consistency_loss=extended_consistency_loss
            * (epoch >= consistency_ignite_epoch),
            true_latents=true_latents,
            true_figures=true_figures,
        )

        if "loss" in output_dict:
            model_loss = output_dict["loss"]
            accum_model_loss += model_loss.item() * accum_adjustment

        # calculate reconstruction loss for all models with the decoder
        reconstruction_loss = metrics.reconstruction_loss(
            images, output_dict["reconstructed_image"]
        )
        accum_reconstruction_loss += reconstruction_loss.item() * accum_adjustment

        reconstruction_r2 = metrics.image_r2_score(
            images.clone(), output_dict["reconstructed_image"]
        )
        accum_reconstruction_r2 += reconstruction_r2.item() * accum_adjustment

        if (
            model.model_name not in ["SlotMLPAdditive", "SlotMLPMonolithic"]
            and epoch % freq == 0
        ):
            true_masks = training_utils.get_masks(images, true_figures)
            ari_score = metrics.ari(
                true_masks,
                output_dict["reconstructed_masks"].detach().permute(1, 0, 2, 3, 4),
            )
            true_masks = true_masks.detach().permute(1, 0, 2, 3, 4)
            accum_ari_score += ari_score.item() * accum_adjustment

        if model.model_name in ["monet", "genesis"]:
            reconstruction_loss = model_loss
        # add to total loss
        total_loss += reconstruction_loss * reconstruction_term_weight

        # calculate slots loss and r2 score for supervised models
        if not unsupervised_mode:
            slots_loss, inds = metrics.hungarian_slots_loss(
                true_latents,
                output_dict["predicted_latents"],
                device,
            )
            accum_slots_loss += slots_loss.item() * accum_adjustment

            avg_r2, raw_r2 = metrics.r2_score(
                true_latents, output_dict["predicted_latents"], inds
            )
            accum_r2_score += avg_r2 * accum_adjustment
            per_latent_r2_score += raw_r2 * accum_adjustment

            # add to total loss
            total_loss += slots_loss

        # calculate consistency loss
        if model.model_name != "SlotMLPMonolithic":
            with nullcontext() if use_consistency_loss else torch.no_grad():
                consistency_encoder_loss, _ = metrics.hungarian_slots_loss(
                    output_dict["sampled_latents"],
                    output_dict["predicted_sampled_latents"],
                    device,
                )
                accum_consistency_encoder_loss += (
                    consistency_encoder_loss.item() * accum_adjustment
                )

            consistency_loss = (
                consistency_encoder_loss
                * consistency_encoder_term_weight
                * use_consistency_loss
            )

        if model.model_name != "SlotMLPMonolithic":
            with nullcontext() if extended_consistency_loss else torch.no_grad():
                consistency_decoder_loss = metrics.reconstruction_loss(
                    output_dict["sampled_image"],
                    output_dict["reconstructed_sampled_image"],
                )
                accum_consistency_decoder_loss += (
                    consistency_decoder_loss.item() * accum_adjustment
                )

        # add to consistency loss only if extended_consistency_loss is True
        if extended_consistency_loss:
            consistency_loss += (
                consistency_decoder_loss
                * consistency_decoder_term_weight
                * extended_consistency_loss
            )

        if model.model_name != "SlotMLPMonolithic":
            consistency_loss *= consistency_term_weight
            accum_consistency_loss += consistency_loss.item() * accum_adjustment

        if (
            use_consistency_loss or extended_consistency_loss
        ) and epoch >= consistency_ignite_epoch:
            # add to total loss
            total_loss += consistency_loss

        accum_total_loss += total_loss.item() * accum_adjustment
        if mode == "train":
            total_loss.backward()
            optimizer.step()

    if model.model_name in ["monet", "genesis"]:
        accum_total_loss -= accum_model_loss
        accum_total_loss += accum_model_loss * dataloader.batch_size
    # logging utils
    training_utils.print_metrics_to_console(
        epoch,
        accum_total_loss,
        accum_reconstruction_loss,
        accum_consistency_loss,
        accum_r2_score,
        accum_slots_loss,
        accum_consistency_encoder_loss,
        accum_consistency_decoder_loss,
    )
    # if epoch % freq == 0:
    #     print(f"ARI score: {accum_ari_score:.4f}")
    #     wandb_utils.wandb_log(
    #         data_path,
    #         kwargs["dataset_name"],
    #         **output_dict,
    #         **locals(),
    #     )

    return accum_reconstruction_loss


def run(
    *,
    model_name,
    dataset_name,
    device,
    epochs,
    batch_size,
    lr,
    lr_scheduler_step,
    reconstruction_term_weight,
    consistency_term_weight,
    consistency_encoder_term_weight,
    consistency_decoder_term_weight,
    consistency_ignite_epoch,
    use_consistency_loss,
    extended_consistency_loss,
    unsupervised_mode,
    n_samples_train,
    n_samples_truncate,
    n_samples_test,
    n_slots,
    n_slot_latents,
    no_overlap,
    sample_mode_train,
    sample_mode_test_id,
    sample_mode_test_ood,
    delta,
    seed,
    load_checkpoint,
    save_name,
    test_freq=20,
):
    """
    Run the training and testing. Currently only supports SpritesWorld dataset.
    Check main.py for the description of the parameters.
    """
    global data_path
    data_path = os.path.join(data_utils.data_path, dataset_name)

    signature_args = locals().copy()
    # wandb_config = signature_args
    # wandb.init(
    #     config=wandb_config,
    #     project="object_centric_ood",
    #     dir=os.path.join(data_utils.data_path, "wandb"),
    # )

    # for mode in ["ID", "OOD", "RDM"]:
    #     wandb.define_metric(f"test_{mode} reconstruction loss", summary="min")
    # wandb.define_metric("ID_score_ID", summary="max")
    # wandb.define_metric("ID_score_OOD", summary="max")
    # for mode in ["ID", "OOD", "RDM"]:
    #     wandb.define_metric(f"test_{mode} ari score", summary="max")
    # wandb_utils.wandb_log_code(wandb.run)

    time_created = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    training_utils.set_seed(seed)

    ##### Loading Data #####
    os.path.isdir(data_path)
    wrapper = src.datasets.wrappers.get_wrapper(
        dataset_name,
        path=data_path,
        load=True,
        save=False,
    )

    test_loader_id = wrapper.get_test_loader(
        sample_mode_test=sample_mode_test_id, **signature_args
    )
    test_loader_ood = wrapper.get_test_loader(
        sample_mode_test=sample_mode_test_ood, **signature_args
    )
    train_loader = wrapper.get_train_loader(**signature_args)

    min_reconstruction_loss_ID = float("inf")
    min_reconstruction_loss_OOD = float("inf")

    in_channels = 3
    if dataset_name == "dsprites":
        resolution = (64, 64)
        ch_dim = 32  # originally 32
    elif dataset_name == "kubric":
        resolution = (128, 128)
        ch_dim = 64  # originally 64

    if model_name == "SlotMLPAdditive":
        model = base_models.SlotMLPAdditive(
            in_channels,
            n_slots,
            n_slot_latents,
            no_overlap=no_overlap,
            dataset_name=dataset_name,
        ).to(device)
    elif model_name == "SlotMLPMonolithic":
        model = base_models.SlotMLPMonolithic(
            in_channels, n_slots, n_slot_latents, dataset_name=dataset_name
        ).to(device)
    elif model_name == "SlotAttention":
        encoder = slot_attention.SlotAttentionEncoder(
            resolution=resolution,
            hid_dim=n_slot_latents,
            ch_dim=ch_dim,
            dataset_name=dataset_name,
        ).to(device)
        decoder = slot_attention.SlotAttentionDecoder(
            hid_dim=n_slot_latents,
            ch_dim=ch_dim,
            resolution=resolution,
            dataset_name=dataset_name,
        ).to(device)
        model = slot_attention.SlotAttentionAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            num_slots=n_slots,
            num_iterations=3,
            hid_dim=n_slot_latents,
            dataset_name=dataset_name,
            no_overlap=no_overlap,
            sampling=True, # change to False for the fixed model
            softmax=True, # change to False for the fixed model
        ).to(device)
    elif model_name == "MONet":
        model = src.models.get_monet_model(n_slots, n_slot_latents, device)
    elif model_name == "GENESIS":
        model = src.models.get_genesis_model(n_slots, n_slot_latents, device)

    # wandb.watch(model)

    # warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)

    start_epoch = 0
    if load_checkpoint:
        model, optimizer, scheduler, start_epoch = training_utils.load_checkpoint(
            model, optimizer, scheduler, load_checkpoint
        )
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 2

    start_epoch += 1

    for epoch in range(start_epoch, epochs + 1):
        if epoch == consistency_ignite_epoch and use_consistency_loss:
            training_utils.save_checkpoint(
                path=data_utils.data_path,
                **locals(),
                checkpoint_name=f"before_ignite_model_{sample_mode_train}_{seed}",
            )
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 2

        rec_loss = one_epoch(
            model,
            train_loader,
            optimizer,
            mode="train",
            epoch=epoch,
            **signature_args,
        )

        if scheduler.get_last_lr()[0] >= 1e-7:
            scheduler.step()

        if scheduler.get_last_lr()[0] > lr:
            optimizer.param_groups[0]["lr"] = lr
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_scheduler_step, gamma=0.5
            )

        print("Learning rate: ", optimizer.param_groups[0]["lr"])

        if epoch % 1 == 0:
            if (
                model_name in ["SlotAttention", "SlotMLPAdditive", "MONet", "GENESIS"]
                and epoch % 1 == 0
            ):
                if dataset_name == "dsprites":
                    categorical_dimensions = [2]
                elif dataset_name == "kubric":
                    categorical_dimensions = [5]

                id_score_id, id_score_ood = metrics.identifiability_score(
                    model,
                    test_loader_id,
                    test_loader_ood,
                    categorical_dimensions,
                    device,
                )
                # wandb.log(
                #     {
                #         "ID_score_ID": id_score_id,
                #         "ID_score_OOD": id_score_ood,
                #     },
                #     step=epoch,
                # )

            id_rec_loss = one_epoch(
                model,
                test_loader_id,
                optimizer,
                mode="test_ID",
                epoch=epoch,
                **signature_args,
            )

            ood_rec_loss = one_epoch(
                model,
                test_loader_ood,
                optimizer,
                mode="test_OOD",
                epoch=epoch,
                **signature_args,
            )

            checkpoint_save_path = data_utils.data_path # change this to your path
            training_utils.save_checkpoint(
                path=checkpoint_save_path,
                **locals(),
                checkpoint_name=f"{save_name}_{epoch}",
            )

    training_utils.save_checkpoint(
        path=checkpoint_save_path,
        **locals(),
        checkpoint_name=f"{save_name}_{epoch}",
    )
