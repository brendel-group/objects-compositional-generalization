import os

import src.datasets.configs as data_configs
import src.datasets.utils as data_utils
import torch
import torchvision
import wandb
from notebooks import utils


def __log_images(log_dict, images, title):
    show_pred_img = images[:8, ...].cpu().clamp(0, 1)
    img_grid = torchvision.utils.make_grid(show_pred_img, pad_value=1)
    log_dict[title] = [wandb.Image(img_grid)]


def __log_figures(log_dict, figures, title):
    for i, figure in enumerate(figures):
        show_pred_img = figure.squeeze(0)[:8, ...].cpu().clamp(0, 1)
        img_grid = torchvision.utils.make_grid(show_pred_img, pad_value=1)
        log_dict[f"{title} figure {i}"] = [wandb.Image(img_grid)]


def wandb_log(
    data_path,
    dataset_name,
    *,
    model,
    mode,
    epoch,
    freq,
    accum_total_loss=None,
    accum_slots_loss=None,
    r2_score=None,
    accum_ari_score=None,
    per_latent_r2_score=None,
    accum_reconstruction_loss=None,
    accum_reconstruction_r2=None,
    images=None,
    reconstructed_image=None,
    reconstructed_figures=None,
    true_masks=None,
    reconstructed_masks=None,
    raw_sampled_figures=None,
    raw_figures=None,
    accum_consistency_encoder_loss=None,
    accum_consistency_decoder_loss=None,
    sampled_image=None,
    sampled_figures=None,
    sampled_masks=None,
    reconstructed_sampled_image=None,
    **kwargs,
):
    if type(per_latent_r2_score) == int:
        per_latent_r2_score = None

    log_dict = {}
    if reconstructed_image is not None and epoch % freq == 0:
        __log_images(log_dict, reconstructed_image, f"{mode} reconstruction")

    if reconstructed_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, reconstructed_figures, f"{mode}")

    if reconstructed_masks is not None and epoch % freq == 0:
        __log_figures(log_dict, reconstructed_masks, f"{mode} masks")

    if images is not None and epoch % freq == 0:
        __log_images(log_dict, images, f"{mode} target")

    if accum_reconstruction_loss is not None:
        log_dict[f"{mode} reconstruction loss"] = accum_reconstruction_loss

    if accum_reconstruction_r2 is not None:
        log_dict[f"{mode} reconstruction r2"] = accum_reconstruction_r2

    if accum_slots_loss is not None:
        log_dict[f"{mode} slots loss"] = accum_slots_loss

    if r2_score is not None:
        log_dict[f"{mode} r2 score"] = r2_score

    if accum_ari_score is not None:
        log_dict[f"{mode} ari score"] = accum_ari_score

    if per_latent_r2_score is not None:
        for i, latent_r2 in enumerate(per_latent_r2_score):
            log_dict[f"{mode} latent {i} r2 score"] = latent_r2

    if accum_total_loss is not None:
        log_dict[f"{mode} total loss"] = accum_total_loss

    if accum_consistency_encoder_loss is not None:
        log_dict[f"{mode} consistency encoder loss"] = accum_consistency_encoder_loss

    if accum_consistency_decoder_loss is not None:
        log_dict[f"{mode} consistency decoder loss"] = accum_consistency_decoder_loss

    if sampled_image is not None and epoch % freq == 0:
        __log_images(log_dict, sampled_image, f"{mode} sampled")

    if sampled_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, sampled_figures, f"{mode} sampled")

    if raw_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, raw_figures, f"{mode} raw")

    if raw_sampled_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, raw_sampled_figures, f"{mode} raw sampled")

    if sampled_masks is not None and epoch % freq == 0:
        __log_figures(log_dict, sampled_masks, f"{mode} sampled masks")

    if true_masks is not None and epoch % freq == 0:
        __log_figures(log_dict, true_masks, f"{mode} true masks")

    if reconstructed_sampled_image is not None and epoch % freq == 0:
        __log_images(
            log_dict, reconstructed_sampled_image, f"{mode} sampled reconstruction"
        )

    wandb.log(log_dict, step=epoch)


def wandb_log_code(run):
    print(os.getcwd())
    if os.path.exists(data_utils.code_path):
        print(os.listdir(data_utils.code_path))
        run.log_code(
            root=data_utils.code_path,
            include_fn=lambda path: any(
                path.endswith(ending) for ending in [".py", ".yaml"]
            ),
        )
    else:
        print("Code path:", data_utils.code_path)
        print("Code path does not exist! Code not logged to wandb.")
