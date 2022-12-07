import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import R2Score

import torchvision

writer = SummaryWriter()

from . import config
from . import data
from . import models


def calculate_r2_score(true_latents, predicted_latents, inds):
    """Calculate R2 score. Slots are flattened before calculating R2 score."""
    inds = torch.LongTensor(inds)
    predicted_latents = predicted_latents.detach().cpu()
    true_latents = true_latents.detach().cpu()

    for i in range(true_latents.shape[0]):
        # shuffling predicted latents to match true latents
        predicted_latents[i, :] = predicted_latents[i, inds[i, :, 1], ...]

    true_latents = true_latents.view(true_latents.shape[0], -1)
    predicted_latents = predicted_latents.reshape(predicted_latents.shape[0], -1)

    r2 = R2Score(true_latents.shape[1])
    return r2(predicted_latents, true_latents)


def matched_slots_loss(
    true_latents, predicted_latents, device="cpu", p=2, reduction="sum"
):
    """
    Computes pairwise distance between slots, matches slots with Hungarian algorithm and outputs
    sum of distances ^ p.
    """
    pairwise_cost = torch.pow(
        torch.cdist(true_latents, predicted_latents, p=p).transpose(-1, -2), p
    )
    indices = np.array(
        list(map(linear_sum_assignment, pairwise_cost.detach().cpu().numpy()))
    )  # applying hungarian algorithm to every sample in batch
    transposed_indices = torch.from_numpy(np.transpose(indices, axes=(0, 2, 1)))

    # extracting the cost of the matched slots; this code is a bit ugly, idk what is the nice way to do it
    output = torch.gather(pairwise_cost, 2, transposed_indices.to(device))[
        :, :, -1
    ].sum(1)

    if reduction == "mean":
        output = output.mean()
    elif reduction == "sum":
        output = output.sum()
    else:
        raise ValueError("Reduction type not supported.")

    return output, transposed_indices


def train(model, train_loader, optimizer, device, epoch=0, reduction="sum"):
    """One epoch of training. Currently only supports MSE loss."""
    model.train()
    train_loss = 0
    total_slots_loss = 0
    total_reconstruction_loss = 0
    r2_score = 0
    for batch_idx, (data, true_latents) in enumerate(train_loader):
        data = data.to(device)
        true_latents = true_latents.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = 0
        if len(output) == 2:
            predicted_images, predicted_latents = output
            loss = F.mse_loss(predicted_images, data, reduction=reduction)
            total_reconstruction_loss += loss.item()
        else:
            predicted_latents = output
        slots_loss, inds = matched_slots_loss(
            predicted_latents, true_latents, device, reduction=reduction
        )
        total_slots_loss += slots_loss.item()

        loss += slots_loss
        loss.backward()
        train_loss += loss.item()
        r2_score += calculate_r2_score(true_latents, predicted_latents, inds)
        optimizer.step()

    print(
        "====> Epoch: {} Average loss: {:.4f}, r2 score {:.4f}".format(
            epoch,
            train_loss / len(train_loader.dataset),
            r2_score / len(train_loader.dataset),
        )
    )
    if len(output) == 2 and epoch % 20 == 0:
        img_grid = torchvision.utils.make_grid(predicted_images.to("cpu"))
        writer.add_image(f"train/predicted", img_grid, epoch)
        img_grid = torchvision.utils.make_grid(data.to("cpu"))
        writer.add_image(f"train/target", img_grid, epoch)

        writer.add_scalar(
            "Average Reconstruction Loss/train",
            total_reconstruction_loss / len(train_loader.dataset),
            epoch,
        )
    writer.add_scalar(
        "Average Slots Loss/train", total_slots_loss / len(train_loader.dataset), epoch
    )
    writer.add_scalar(
        "Average Loss/train", train_loss / len(train_loader.dataset), epoch
    )
    writer.add_scalar(
        "Average R2 Score/train", r2_score / len(train_loader.dataset), epoch
    )


def test(model, test_loader, device, epoch, reduction="sum", test_type="ID"):
    """Test the model on test set. Currently only supports MSE loss."""
    model.eval()
    test_loss = 0
    total_slots_loss = 0
    total_reconstruction_loss = 0
    r2_score = 0
    with torch.no_grad():
        for batch_idx, (data, true_latents) in enumerate(test_loader):
            data = data.to(device)
            true_latents = true_latents.to(device)
            output = model(data)

            loss = 0
            if len(output) == 2:
                predicted_images, predicted_latents = output
                loss = F.mse_loss(predicted_images, data, reduction=reduction)
                total_reconstruction_loss += loss.item()
            else:
                predicted_latents = output
            slots_loss, inds = matched_slots_loss(
                predicted_latents, true_latents, device, reduction=reduction
            )
            total_slots_loss += slots_loss.item()

            loss += slots_loss
            r2_score += calculate_r2_score(true_latents, predicted_latents, inds)
            test_loss += loss.item()

    print(
        "===========> {} Test set loss: {:.4f}, r2 score {:.4f}".format(
            test_type,
            test_loss / len(test_loader.dataset),
            r2_score / len(test_loader.dataset),
        ),
    )
    if len(output) == 2:
        img_grid = torchvision.utils.make_grid(predicted_images.to("cpu"))
        writer.add_image(f"test_{test_type}/predicted", img_grid, epoch)
        img_grid = torchvision.utils.make_grid(data.to("cpu"))
        writer.add_image(f"test_{test_type}/target", img_grid, epoch)

        writer.add_scalar(
            f"Average Reconstruction Loss/test_{test_type}",
            total_reconstruction_loss / len(test_loader.dataset),
            epoch,
        )
    writer.add_scalar(
        f"Average Slots Loss/test_{test_type}",
        total_slots_loss / len(test_loader.dataset),
        epoch,
    )
    writer.add_scalar(
        f"Average Loss/test_{test_type}", test_loss / len(test_loader.dataset), epoch
    )
    writer.add_scalar(
        f"Average R2 Score/test_{test_type}", r2_score / len(test_loader.dataset), epoch
    )


def collate_fn_normalizer(batch, bias=0, scale=1):
    """Normalize latents target to [0, 1]."""
    images, latents = zip(*batch)
    latents = torch.stack(latents)
    latents = (latents - bias) / scale
    return torch.stack(images), latents


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cfg = config.SpriteWorldConfig()
    min_offset = torch.FloatTensor(
        [rng.min for rng in cfg.get_ranges().values()]
    ).reshape(1, 1, -1)
    scale = torch.FloatTensor(
        [rng.max - rng.min for rng in cfg.get_ranges().values()]
    ).reshape(1, 1, -1)

    # dataset parameters
    in_channels = 3
    delta = 0.125
    n_slots = 2
    n_slot_latents = 8
    no_overlap = True
    n_samples_train = 10000
    n_samples_test = 1000
    sample_mode_train = "diagonal"
    sample_mode_test_id = "diagonal"
    sample_mode_test_ood = "off_diagonal"

    # training parameters
    lr = 1e-3
    epochs = 2000
    batch_size = 64

    model = models.SlotMLPMonotonic(in_channels, n_slots, n_slot_latents).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # datasets
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = data.SpriteWorldDataset(
        n_samples_test,
        n_slots,
        cfg,
        sample_mode=sample_mode_train,
        delta=delta,
        no_overlap=no_overlap,
        transform=transform,
    )
    test_dataset_id = data.SpriteWorldDataset(
        n_samples_train,
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

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, device, epoch)
        if epoch % 50 == 0:
            test(model, test_loader_id, device, epoch, test_type="ID")
            test(model, test_loader_ood, device, epoch, test_type="OOD")
