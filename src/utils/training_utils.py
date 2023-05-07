import os

import numpy as np
import torch

from typing import Tuple


def sample_z_from_latents(latents):
    """
    Sample "delusional" z samples from latents.

    Args:
        latents: tensor of shape (batch_size, n_slots, n_latents)

    Returns:
        sampled_z: tensor of shape (batch_size, n_slots, n_latents)
    """
    batch_size, n_slots, n_latents = latents.shape

    # Flatten the latents tensor into a 2D tensor
    flattened_latents = latents.view(batch_size * n_slots, n_latents)

    # Sample indices randomly with replacement from the flattened latents tensor
    indices = np.random.choice(
        len(flattened_latents),
        size=batch_size * n_slots,
    )

    # Gather the sampled latents from the flattened tensor
    sampled_latents = flattened_latents[indices]

    # Reshape the sampled latents tensor back to the original shape
    sampled_z = sampled_latents.view(batch_size, n_slots, n_latents)

    return sampled_z.to(latents.device)


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
