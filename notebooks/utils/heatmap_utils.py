import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from torch import nn

from src import config, data

transform = transforms.Compose([transforms.ToTensor()])
default_cfg = config.SpriteWorldConfig()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def plot_dataset_images(dataset, rows=5, cols=5, name="gt", pred=False, id_mask=None):
    fig, ax = plt.subplots(rows, cols, sharex="col", sharey="row", figsize=(10, 10))

    if id_mask:
        id_mtx = np.zeros((rows, cols))
        for (x, y) in list(zip(id_mask[0], id_mask[1])):
            id_mtx[x, y] = 1

    for row in range(rows):
        for col in range(cols):
            if not pred:
                sample = dataset[row * cols + col][0]
            else:
                sample = dataset[row * cols + col]
            sample = sample.permute(1, 2, 0).numpy()

            pad_size = 3
            pad_val = 0
            if id_mask and id_mtx[row, col]:
                pad_val = np.array([195, 40, 40])

            if not pred:
                pad_val = pad_val / 255
            padded = (
                np.zeros(
                    (
                        sample.shape[0] + pad_size * 2,
                        sample.shape[1] + pad_size * 2,
                        sample.shape[2],
                    )
                )
                + pad_val
            )
            padded[pad_size:-pad_size, pad_size:-pad_size, :] = sample

            if pred:
                padded = padded.astype(int)

            ax[row, col].imshow(padded)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{name}.svg")
    plt.show()


def create_traversed_dataset(initial_sample, n_steps=4):
    samples = []
    y_range = np.linspace(0.2, 0.8, n_steps)
    c_0_range = np.linspace(0.1, 0.9, n_steps)

    for i in range(n_steps):
        for j in range(n_steps):
            sample = initial_sample.clone()
            sample[0][1] = y_range[i]
            sample[1][1] = y_range[j]

            sample[0][-3] = c_0_range[i]
            sample[1][-3] = c_0_range[j]

            samples.append(sample)
    traversed_dataset = data.SpriteWorldDataset(
        len(torch.stack(samples)),
        2,
        default_cfg,
        sample_mode="random",
        no_overlap=True,
        delta=0.3,
        z=torch.stack(samples),
        transform=transform,
    )
    return traversed_dataset


def make_pred_and_plot(
    model: nn.Module,
    train_loader,
    device,
    n_steps=4,
    decoder=False,
    name="",
    id_mask=False,
):
    outs = torch.zeros((0, 3, 64, 64))
    for image, label in train_loader:
        if decoder:
            pred_image, _ = model.decoder(label.to(device))
        else:
            pred_image, pred_latents, _ = model(image.to(device))

        pred_image = torch.clip(pred_image, 0, 1) * 255
        outs = torch.vstack([outs, pred_image.cpu().detach()])
    plot_dataset_images(
        outs, rows=n_steps, cols=n_steps, name=name, pred=True, id_mask=id_mask
    )


def get_binary_id_mask(
    dataset: data.SpriteWorldDataset, size: int = 100
) -> (np.array, np.array):
    """Return coordinates of points that are inside the diagonal."""

    def check_distance_from_diag(p3, p1=np.array([0, 0]), p2=np.array([1, 1])):
        """Calculate the distance from a point to a line."""
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        return d

    points = []
    for (x, z) in dataset:
        val = (
            check_distance_from_diag(np.array([z[0][1].item(), z[1][1].item()]))
            <= 0.125
        ) and (
            check_distance_from_diag(np.array([z[0][-3].item(), z[1][-3].item()]))
            <= 0.125
        )
        points.append(int(val))

    mtx_points = np.array(points).reshape(size, size)
    x = []
    y = []
    for i in range(size):
        for j in range(size):
            if mtx_points[i, j]:
                x.append(i)
                y.append(j)
            if mtx_points[i, j]:
                x.append(i)
                y.append(j)
    return np.array(x), np.array(y)


def predict_for_heatmap(model, model_path, data_loader, decoder=True, device="cuda"):
    """Make prediction on traversed images and return loss array."""
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loss_array = torch.Tensor(0, 3, 64, 64)
    for image, label in tqdm.tqdm(data_loader):
        if decoder:
            pred_image, _ = model.decoder(label.to(device))
        else:
            pred_image, latents, _ = model(image.to(device))
        loss_array = torch.cat(
            [loss_array, torch.square(image - pred_image.cpu()).detach()]
        )

    return [i.sum().detach().item() for i in loss_array]
