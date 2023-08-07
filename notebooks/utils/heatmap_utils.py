import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
import tqdm
from torch import nn

from src.datasets import data
import src.datasets.configs as data_configs

default_cfg = data_configs.SpriteWorldConfig()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def plot_dataset_images(dataset, rows=5, cols=5, id_mask=True):
    fig, ax = plt.subplots(rows, cols, figsize=(10, 10))

    if id_mask:
        id_mtx = np.zeros((rows, cols))
        x, y = get_binary_id_mask(dataset, size=5, only_id=True)
        for i, j in list(zip(x, y)):
            id_mtx[i, j] = 1

    for row in range(rows):
        for col in range(cols):
            sample = dataset[row * cols + col][0][-1]
            sample = sample.permute(1, 2, 0).numpy()

            pad_size = 4
            pad_val = 0
            if id_mask and id_mtx[row, col]:
                pad_val = np.array([195, 40, 40])

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

            ax[rows - row - 1, col].imshow(padded)  # Flip rows
            ax[rows - row - 1, col].set_xticks([])
            ax[rows - row - 1, col].set_yticks([])
            for spine in ax[rows - row - 1, col].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"traverse_plot.svg")
    plt.show()


def create_traversed_dataset(initial_sample, n_steps=4):
    # Define the ranges for y and c_0
    y_range = torch.linspace(0.2, 0.8, n_steps)
    c_0_range = torch.linspace(0.05, 0.95, n_steps)

    y_combinations = torch.cartesian_prod(y_range, y_range)
    c_0_combinations = torch.cartesian_prod(c_0_range, c_0_range)

    # Create a tensor of repeated initial samples with the shape (n_steps, initial_sample[0], initial_sample[1])
    samples = initial_sample.repeat((n_steps**2, 1, 1))

    samples[:, 0, 1] = y_combinations[:, 0]
    samples[:, 1, 1] = y_combinations[:, 1]
    samples[:, 0, -3] = c_0_combinations[:, 0]
    samples[:, 1, -3] = c_0_combinations[:, 1]

    # Create a dataset using the tensor of samples
    traversed_dataset = data.SpriteWorldDataset(
        len(samples),
        2,
        default_cfg,
        sample_mode="skip",
        no_overlap=True,
        delta=0.125,
        z=samples,
        transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
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
            output = model.decoder(label.to(device))
            pred_image = output[0]
        else:
            output = model(image.to(device))
            pred_image, pred_latents = output[0], output[1]

        pred_image = torch.clip(pred_image, 0, 1) * 255
        outs = torch.vstack([outs, pred_image.cpu().detach()])
    plot_dataset_images(
        outs, rows=n_steps, cols=n_steps, name=name, pred=True, id_mask=id_mask
    )


def get_binary_id_mask(
    dataset: data.SpriteWorldDataset, size: int = 100, only_id: bool = False
) -> (np.array, np.array):
    """Return coordinates of points that are inside the diagonal."""

    def check_distance_from_diag(p3, p1=np.array([0, 0]), p2=np.array([1, 1])):
        """Calculate the distance from a point to a line."""
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        return d

    points = []
    for x, z in dataset:
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

    if only_id:
        return x, y
    # code for outlining ID samples region
    id_mtx = np.zeros((100, 100))
    for i, j in list(zip(x, y)):
        id_mtx[i, j] = 1

    x_line_left = []
    y_line_left = []
    x_line_right = [0.5]
    y_line_right = [0.5]
    for i in range(100):
        for j in range(100):
            if j != 99 and id_mtx[i, j] and id_mtx[i, j + 1] == 0:
                x_line_right.append(i + 0.5)
                y_line_right.append(j + 0.5)
            if id_mtx[i, j] and id_mtx[i, j - 1] == 0:
                x_line_left.append(i + 0.5)
                y_line_left.append(j + 0.5)

    x_line_right.append(70 + 0.5)
    y_line_right.append(99 + 0.5)
    x_line_right.append(99 + 0.5)
    y_line_right.append(99 + 0.5)
    x_line_left.append(99 + 0.5)
    y_line_left.append(99 + 0.5)

    x_line_right = np.array(x_line_right)
    y_line_right = np.array(y_line_right)
    x_line_left = np.array(x_line_left)
    y_line_left = np.array(y_line_left)
    return x_line_right, y_line_right, x_line_left, y_line_left


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


def get_id_bounds(x, y, shape):
    id_mtx = np.zeros((shape + 2, shape + 2))
    id_mtx[(x + 1, y + 1)] = 1

    x_line_left, y_line_left, x_line_right, y_line_right = [], [], [1], [1]
    for i in range(id_mtx.shape[0]):
        for j in range(id_mtx.shape[1]):
            if id_mtx[i, j]:
                if not id_mtx[i, j + 1]:
                    x_line_right.append(i)
                    y_line_right.append(j)
                if not id_mtx[i, j - 1]:
                    x_line_left.append(i)
                    y_line_left.append(j)

    x_line_left += [id_mtx.shape[0] - 2]
    y_line_left += [id_mtx.shape[0] - 2]
    x_line_right += [id_mtx.shape[1] - 2]
    y_line_right += [id_mtx.shape[1] - 2]

    return (
        np.array(x_line_left) - 0.5,
        np.array(y_line_left) - 0.5,
        np.array(x_line_right) - 0.5,
        np.array(y_line_right) - 0.5,
    )
