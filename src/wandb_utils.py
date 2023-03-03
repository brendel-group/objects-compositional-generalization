import wandb
import torchvision


def __log_images(log_dict, images, title):
    show_pred_img = images[:8, ...].cpu().clamp(0, 1)
    img_grid = torchvision.utils.make_grid(show_pred_img, pad_value=1)
    log_dict[title] = [wandb.Image(img_grid)]


def __log_figures(log_dict, figures, title):
    for i, figure in enumerate(figures):
        show_pred_img = figure[:8, ...].cpu().clamp(0, 1)
        img_grid = torchvision.utils.make_grid(show_pred_img)
        log_dict[f"{title} figure {i}"] = [wandb.Image(img_grid)]


def wandb_log(
    mode,
    epoch,
    freq,
    *,
    total_loss=None,
    slots_loss=None,
    r2_score=None,
    r2_score_raw=None,
    reconstruction_loss=None,
    true_images=None,
    predicted_images=None,
    predicted_figures=None,
    consistency_loss=None,
    sampled_images=None,
    sampled_figures=None,
):
    if type(r2_score_raw) == int:
        r2_score_raw = None

    log_dict = {}
    if predicted_images is not None and epoch % freq == 0:
        __log_images(log_dict, predicted_images, f"{mode} reconstruction")

    if predicted_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, predicted_figures, f"{mode}")

    if true_images is not None and epoch % freq == 0:
        __log_images(log_dict, true_images, f"{mode} target")

    if reconstruction_loss is not None:
        log_dict[f"{mode} reconstruction loss"] = reconstruction_loss

    if slots_loss is not None:
        log_dict[f"{mode} slots loss"] = slots_loss

    if r2_score is not None:
        log_dict[f"{mode} r2 score"] = r2_score

    if r2_score_raw is not None:
        for i, latent_r2 in enumerate(r2_score_raw):
            log_dict[f"{mode} latent {i} r2 score"] = latent_r2

    if total_loss is not None:
        log_dict[f"{mode} total loss"] = total_loss

    if consistency_loss is not None:
        log_dict[f"{mode} consistency loss"] = consistency_loss

    if sampled_images is not None and epoch % freq == 0:
        __log_images(log_dict, sampled_images, f"{mode} sampled")

    if sampled_figures is not None and epoch % freq == 0:
        __log_figures(log_dict, sampled_figures, f"{mode} sampled")

    wandb.log(log_dict, step=epoch)
