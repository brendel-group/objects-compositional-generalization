import os

import torch
import torchvision

import wandb

import src.utils.data_utils
from notebooks import utils
from src import config
from src.utils import data_utils, training_utils


def __log_images(log_dict, images, title):
    show_pred_img = images[:8, ...].cpu().clamp(0, 1)
    img_grid = torchvision.utils.make_grid(show_pred_img, pad_value=1)
    log_dict[title] = [wandb.Image(img_grid)]


def __log_figures(log_dict, figures, title):
    for i, figure in enumerate(figures):
        show_pred_img = figure[:8, ...].cpu().clamp(0, 1)
        img_grid = torchvision.utils.make_grid(show_pred_img, pad_value=1)
        log_dict[f"{title} figure {i}"] = [wandb.Image(img_grid)]


def wandb_log(
    *,
    model,
    mode,
    epoch,
    freq,
    accum_total_loss=None,
    accum_slots_loss=None,
    r2_score=None,
    per_latent_r2_score=None,
    accum_reconstruction_loss=None,
    images=None,
    predicted_images=None,
    predicted_figures=None,
    accum_consistency_encoder_loss=None,
    accum_consistency_decoder_loss=None,
    sampled_images=None,
    sampled_figures=None,
    **kwargs,
):
    if type(per_latent_r2_score) == int:
        per_latent_r2_score = None

    log_dict = {}
    if predicted_images is not None and epoch % freq == 0:
        __log_images(log_dict, predicted_images, f"{mode} reconstruction")

    if predicted_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, predicted_figures, f"{mode}")

    if images is not None and epoch % freq == 0:
        __log_images(log_dict, images, f"{mode} target")

    if accum_reconstruction_loss is not None:
        log_dict[f"{mode} reconstruction loss"] = accum_reconstruction_loss

    if accum_slots_loss is not None:
        log_dict[f"{mode} slots loss"] = accum_slots_loss

    if r2_score is not None:
        log_dict[f"{mode} r2 score"] = r2_score

    if per_latent_r2_score is not None:
        for i, latent_r2 in enumerate(per_latent_r2_score):
            log_dict[f"{mode} latent {i} r2 score"] = latent_r2

    if accum_total_loss is not None:
        log_dict[f"{mode} total loss"] = accum_total_loss

    if accum_consistency_encoder_loss is not None:
        log_dict[f"{mode} consistency encoder loss"] = accum_consistency_encoder_loss

    if accum_consistency_decoder_loss is not None:
        log_dict[f"{mode} consistency decoder loss"] = accum_consistency_decoder_loss

    if sampled_images is not None and epoch % freq == 0:
        __log_images(log_dict, sampled_images, f"{mode} sampled")

    if sampled_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, sampled_figures, f"{mode} sampled")

    if epoch % (freq * 2) == 0:
        __make_histogram(log_dict, model, f"Heatmap (wandb)")

    wandb.log(log_dict, step=epoch)


def wandb_log_code(run):
    print(os.getcwd())
    run.log_code(
        "/home/bethge/apanfilov27/tmp/object_centric_ood/src/",
        include_fn=lambda path: path.endswith(".py"),
    )


def __make_histogram(log_dict, model, title):
    n_steps = 50
    initial_sample = torch.tensor(
        [
            [0.4504, 0.5, 0, 0.1, 0.0000, 0.42, 1, 1],
            [0.7401, 0.5, 0, 0.1, 0.0000, 0.38, 1, 1],
        ]
    )
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

    path = "/mnt/qb/work/bethge/apanfilov27"
    if os.path.isdir(os.path.join(path, "traversed")):
        traversed_dataset = data_utils.PreGeneratedDataset(
            os.path.join(path, "traversed")
        )
        print("Traversed dataset successfully loaded from disk.")
    else:
        traversed_dataset = utils.create_traversed_dataset(
            initial_sample, n_steps=n_steps
        )
        data_utils.dump_generated_dataset(
            traversed_dataset, os.path.join(path, "traversed")
        )

    heatmap_loader = torch.utils.data.DataLoader(
        traversed_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda b: src.utils.data_utils.collate_fn_normalizer(
            b, min_offset, scale
        ),
    )

    loss_array = torch.Tensor(0, 3, 64, 64)
    model.eval()
    for image, _ in heatmap_loader:
        image = image.to("cuda")
        output = model(image)
        pred_image, latents = output[0], output[1]

        loss_array = torch.cat(
            [loss_array, torch.square(image.cpu() - pred_image.cpu()).detach()]
        )

    loss_array = [i.sum().detach().item() for i in loss_array]
    x, y = utils.get_binary_id_mask(traversed_dataset, n_steps)
    x_line_left, y_line_left, x_line_right, y_line_right = utils.get_id_bounds(
        x, y, shape=n_steps
    )

    utils.plot_heatmap(
        [loss_array],
        (x_line_left, y_line_left),
        (x_line_right, y_line_right),
        shape=n_steps,
        save_name=f"{wandb.run.name}.png",
        figsize=(6, 5),
        show=False,
    )

    log_dict[f"{title}"] = [wandb.Image(f"heatmaps/{wandb.run.name}.png")]
