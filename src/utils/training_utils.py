import os

import numpy as np
import torch

from typing import Tuple


def sample_z_from_latents(latents, n_samples=64):
    """
    Sample "delusional" z samples from latents.

    Args:
        latents: tensor of shape (batch_size, n_slots, n_latents)
        n_samples: number of samples to generate

    Returns:
        sampled_z: tensor of shape (n_samples, n_slots, n_latents)
        indices: array of indices of the sampled latents
    """
    batch_size, n_slots, n_latents = latents.shape

    # Flatten the latents tensor into a 2D tensor
    flattened_latents = latents.reshape(batch_size * n_slots, n_latents)

    # Sample indices randomly with replacement from the flattened latents tensor
    indices = np.random.choice(
        len(flattened_latents),
        size=n_samples * n_slots,
    )

    # Gather the sampled latents from the flattened tensor
    sampled_latents = flattened_latents[indices]

    # Reshape the sampled latents tensor back to the original shape
    sampled_z = sampled_latents.reshape(n_samples, n_slots, n_latents)

    return sampled_z.to(latents.device), indices


def get_masks(x, figures, threshold=0.1):
    """
    Get the masks of the objects in the images, given the original image and per-object images.
    Args:
        x: Tensor of shape [batch_size, n_ch, res_x, res_y] containing the original images.
        figures: Tensor of shape [batch_size, n_slots, n_ch, res_x, res_y] containing the object images.
        threshold: Threshold for the mask (removes some noise artifacts).
    Returns:
        masks: Tensor of shape [batch_size, n_slots, n_ch, res_x, res_y] containing the masks of the objects in the images.
    """
    # Prepare the mask tensor
    masks = torch.zeros(
        (
            figures.shape[0],
            figures.shape[1] + 1,
            1,
            figures.shape[3],
            figures.shape[4],
        )  # +1 for background
    ).to(x.device)

    # Iterate over the original images
    for i in range(x.shape[0]):
        background_mask = torch.ones(x[i].shape).bool().to(x.device)
        for j in range(0, figures.shape[1]):
            figure_mask = torch.isclose(
                x[i],
                figures[i, j],
                atol=threshold,
            )
            background_mask = background_mask * figure_mask

        background_mask = ~background_mask
        background_mask = background_mask.float().mean(dim=0, keepdim=True)
        background_mask = (background_mask > 0).float()

        for j in range(figures.shape[1]):
            # Get the figure
            figure = figures[i, j]  # [n_ch, res_x, res_y]

            # Get the mask
            mask = (
                torch.isclose(
                    x[i],
                    figure,
                    atol=threshold,
                )
                .float()
                .mean(dim=0, keepdim=True)
            )
            mask = (mask == 1.0).float()

            mask *= background_mask

            masks[i, j + 1] = mask

        # Get the background mask
        background_mask = (~background_mask.bool()).float()
        masks[i, 0] = background_mask

    return masks


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_metrics_to_console(
    epoch: int,
    accum_total_loss: float,
    accum_reconstruction_loss: float,
    accum_consistency_loss: float,
    accum_r2_score: float,
    accum_slots_loss: float,
    accum_encoder_consistency_loss: float,
    accum_decoder_consistency_loss: float,
) -> None:
    """
    Prints the metrics to console in a tabular format.

    Args:
    epoch: The current epoch.
    accum_total_loss: The accumulated total loss.
    accum_reconstruction_loss: The accumulated reconstruction loss.
    accum_consistency_loss: The accumulated consistency loss.
    accum_r2_score: The accumulated R2 score.
    accum_slots_loss: The accumulated slots loss.
    accum_encoder_consistency_loss: The accumulated encoder consistency loss.
    accum_decoder_consistency_loss: The accumulated decoder consistency loss.
    """
    print(f"{'=' * 130}")
    print(
        f"{'Epoch':<10}{'Avg. loss (Rec. + Cons. weighted)':<40}{'Cons. encoder loss':<25}{'Cons. decoder loss':<25}{'R2 score':<15}{'Slots loss':<15}"
    )
    print(f"{'=' * 130}")
    print(
        "{:<10}{:<40}{:<25}{:<25}{:<15}{:<15}".format(
            " " + str(epoch),
            "{:.4f} = {:.4f} + {:.4f}".format(
                accum_total_loss, accum_reconstruction_loss, accum_consistency_loss
            ),
            "{:.4f}".format(accum_encoder_consistency_loss),
            "{:.4f}".format(accum_decoder_consistency_loss),
            "{:.4f}".format(accum_r2_score),
            "{:.4f}".format(accum_slots_loss),
            "",
            "",
        )
    )
    print(f"{'=' * 130}")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    model_name,
    epoch,
    time_created,
    checkpoint_name,
    path,
    **kwargs,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(
        checkpoint,
        os.path.join(
            path,
            "checkpoints",
            f"{model_name}_{time_created}_{checkpoint_name}_checkpoint.pt",
        ),
    )


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, scheduler, epoch
