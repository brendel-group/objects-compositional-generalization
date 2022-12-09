import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import torchvision

from .training_utils import (
    calculate_r2_score,
    matched_slots_loss,
    collate_fn_normalizer,
)

writer = SummaryWriter()

from . import config
from . import data
from . import models


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
            total_reconstruction_loss += loss.item() * len(data)
        else:
            predicted_latents = output
        slots_loss, inds = matched_slots_loss(
            predicted_latents, true_latents, device, reduction=reduction
        )
        total_slots_loss += slots_loss.item() * len(data)

        loss += slots_loss
        loss.backward()
        train_loss += loss.item() * len(data)
        r2_score += calculate_r2_score(true_latents, predicted_latents, inds) * len(data)
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
                total_reconstruction_loss += loss.item() * len(data)
            else:
                predicted_latents = output
            slots_loss, inds = matched_slots_loss(
                predicted_latents, true_latents, device, reduction=reduction
            )
            total_slots_loss += slots_loss.item() * len(data)

            loss += slots_loss
            r2_score += calculate_r2_score(true_latents, predicted_latents, inds) * len(data)
            test_loss += loss.item() * len(data)

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

    model = models.SlotMLPMonolithic(in_channels, n_slots, n_slot_latents).to(device)
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
