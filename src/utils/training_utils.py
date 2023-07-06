import os

import numpy as np
import torch

from src.metrics import hungarian_slots_loss


def sample_z_from_latents_no_overlap(
    gt_z, hat_z, gt_figures, hat_figures, device, n_samples=64
):
    _, transposed_indices = hungarian_slots_loss(
        gt_figures.view(gt_figures.shape[0], gt_figures.shape[1], -1),
        hat_figures.view(hat_figures.shape[0], hat_figures.shape[1], -1),
        device=device,
    )

    transposed_indices = transposed_indices.to(device)

    hat_z_permuted = hat_z.gather(
        1,
        transposed_indices[:, :, 1].unsqueeze(-1).expand(-1, -1, hat_z.shape[-1]),
    )
    gt_z_flatten = gt_z.view(-1, gt_z.shape[2])
    z_sampled, indices = sample_z_from_latents(hat_z_permuted.detach(), n_samples=1024)

    # reshape z_flatten with indices
    z_flatten = gt_z_flatten[indices].reshape(-1, gt_z.shape[1], gt_z.shape[2])

    z_flatten_sampled, selected_pairs_indices = filter_objects(
        z_flatten, max_objects=n_samples
    )
    z_sampled = z_sampled[selected_pairs_indices]
    return z_sampled


def filter_objects(latents, max_objects=5000, threshold=0.2, sort=False):
    """
    Filter objects based on their Euclidean distance.
    Args:
        latents: Tensor of shape (batch_size, n_slots, n_latents)
        max_objects: Number of objects to keep at most
        threshold: Distance threshold
        sort: Whether to sort the objects by distance
    """
    N, slots, _ = latents.size()
    mask = torch.zeros(N, dtype=bool)

    # Compute Euclidean distance for each pair of slots in each item
    for n in range(N):
        slots_distances = torch.cdist(latents[n, :, :2], latents[n, :, :2], p=2)
        slots_distances.fill_diagonal_(float('inf'))  # Ignore distance to self

        # Consider an object as "close" if its minimal distance to any other object is below the threshold
        min_distance = slots_distances.min().item()
        if min_distance >= threshold:
            mask[n] = True

    # If all objects are "close", print a message and return
    if not torch.any(mask):
        print("No objects were found that meet the distance threshold.")
        return None, []

    # Apply the mask to the latents
    filtered_objects = latents[mask]
    filtered_indices = torch.arange(N)[mask]

    # If the number of filtered objects exceeds the maximum, truncate them
    if filtered_objects.size(0) > max_objects:
        filtered_objects = filtered_objects[:max_objects]
        filtered_indices = filtered_indices[:max_objects]

    if sort:
        # Sort the filtered objects by minimum distance to any other object
        min_distances = torch.zeros(mask.sum().item())
        for i, n in enumerate(torch.where(mask)[0]):
            slots_distances = torch.cdist(latents[n], latents[n], p=2)
            slots_distances.fill_diagonal_(float('inf'))
            min_distances[i] = slots_distances.min().item()

        indices = torch.argsort(min_distances)
        filtered_objects = filtered_objects[indices]
        filtered_indices = filtered_indices[indices]

    return filtered_objects, filtered_indices.tolist()


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
