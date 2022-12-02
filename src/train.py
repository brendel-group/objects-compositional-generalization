import numpy as np
import torch
import torch.utils.data
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
        predicted_latents[i, :] = predicted_latents[i, inds[i, :, 1], ...]

    true_latents = true_latents.view(true_latents.shape[0], -1)
    predicted_latents = predicted_latents.reshape(predicted_latents.shape[0], -1)

    r2 = R2Score(true_latents.shape[1])
    return r2(true_latents, predicted_latents)


def matched_slots_loss(true_latents, predicted_latents, device="cpu", p=2):
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
    ].sum()
    return output, transposed_indices


def matched_latents_loss(true_latents, predicted_latents, device="cpu", p=2):
    """
    Computes pairwise distance between flattened latents, matches latents with Hungarian algorithm and outputs
    sum of distances ^ p.
    """
    pairwise_cost = torch.pow(
        torch.cdist(
            true_latents.view(true_latents.shape[0], -1, 1),
            predicted_latents.view(predicted_latents.shape[0], -1, 1),
            p=p,
        ).transpose(-1, -2),
        p,
    )
    indices = np.array(
        list(map(linear_sum_assignment, pairwise_cost.detach().cpu().numpy()))
    )  # applying hungarian algorithm to every sample in batch
    transposed_indices = torch.from_numpy(np.transpose(indices, axes=(0, 2, 1)))
    output = torch.gather(pairwise_cost, 2, transposed_indices.to(device))[:, :, -1]
    return output.sum(), transposed_indices


def train(model, train_loader, optimizer, device, epoch=0):
    """One epoch of training. Currently only supports MSE loss."""
    model.train()
    train_loss = 0
    for batch_idx, (data, true_latents) in enumerate(train_loader):
        data = data.to(device)
        true_latents = true_latents.to(device)
        optimizer.zero_grad()
        predicted_latents = model(data)

        if model.model_type in ["slots_regressor"]:
            loss, inds = matched_slots_loss(predicted_latents, true_latents, device)
        elif model.model_type in ["latents_regressor"]:
            loss, inds = matched_latents_loss(predicted_latents, true_latents, device)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )
    writer.add_scalar(
        "Average Loss/train", train_loss / len(train_loader.dataset), epoch
    )


def test(model, test_loader, device, epoch):
    """Test the model on test set. Currently only supports MSE loss."""
    model.eval()
    test_loss = 0
    r2_score = 0
    with torch.no_grad():
        for batch_idx, (data, true_latents) in enumerate(test_loader):
            data = data.to(device)
            true_latents = true_latents.to(device)
            predicted_latents = model(data)
            if model.model_type in ["slots_regressor"]:
                loss, inds = matched_slots_loss(predicted_latents, true_latents, device)
            elif model.model_type in ["latents_regressor"]:
                loss, inds = matched_latents_loss(
                    predicted_latents, true_latents, device
                )
            r2_score += calculate_r2_score(true_latents, predicted_latents, inds)
            test_loss += loss.item()
    print(
        "====> Test set loss: {:.4f}, r2 score {:.4f}".format(
            test_loss / len(test_loader.dataset), r2_score / len(test_loader.dataset)
        ),
    )
    writer.add_scalar("Average Loss/test", test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar("R2 Score/test", r2_score / len(test_loader.dataset), epoch)

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

    delta = 0.3
    n_slots = 2
    no_overlap = True
    n_samples_train = 20000
    n_samples_test = 3000
    sample_mode_train = "diagonal"
    sample_mode_test = "diagonal"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_dataset = data.SpriteWorldDataset(
        n_samples_train,
        n_slots,
        cfg,
        sample_mode=sample_mode_test,
        delta=delta,
        no_overlap=no_overlap,
        transform=transform,
    )

    train_dataset = data.SpriteWorldDataset(
        n_samples_test,
        n_slots,
        cfg,
        sample_mode=sample_mode_train,
        delta=delta,
        no_overlap=no_overlap,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_fn_normalizer(b, min_offset, scale),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate_fn_normalizer(b, min_offset, scale),
    )

    model = models.SlotMLP(3, n_slots, 8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("train_images", img_grid)
    writer.add_graph(model, images.to(device))

    for epoch in range(1, 2000 + 1):
        train(model, train_loader, optimizer, device, epoch)
        if epoch % 50 == 0:
            test(model, test_loader, device, epoch)
