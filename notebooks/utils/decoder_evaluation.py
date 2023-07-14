import sys

# print(sys.path)
sys.path.append("D:\\git_projects\\bethgelab\\lab_rotation\\object_centric_ood")
# print(sys.path)

import numpy as np
import torch
import random

from src.datasets import data, utils, configs
from src.metrics import hungarian_slots_loss, image_r2_score


from torchvision import transforms as transforms

import tqdm

seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cfg = configs.SpriteWorldConfig()

scale = torch.FloatTensor(
    [rng.max - rng.min for rng in cfg.get_ranges().values()]
).reshape(1, 1, -1)
scale = torch.cat([scale[:, :, :-4], scale[:, :, -3:-2]], dim=-1)


min_offset = torch.FloatTensor([rng.min for rng in cfg.get_ranges().values()]).reshape(
    1, 1, -1
)
min_offset = torch.cat([min_offset[:, :, :-4], min_offset[:, :, -3:-2]], dim=-1)


def generate_complementary_x_coordinate(x, delta=0.125, n_slots=2):
    x_diag = torch.Tensor([x, x]) - cfg["x"].min
    x_diag = x_diag / (cfg["x"].max - cfg["x"].min)

    noise = torch.randn(n_slots + 2)
    noise = noise / torch.norm(noise, keepdim=True)
    noise = noise[:n_slots]

    ort_vec = noise - x_diag * torch.dot(noise, x_diag) / torch.dot(x_diag, x_diag)
    ort_vec = ort_vec / torch.norm(ort_vec, keepdim=True)
    # why n - 1 here? because we sample "radius" not in the original space, but in the embedded
    ort_vec *= torch.pow(torch.rand(1), 1 / (n_slots - 1)) * delta
    x_diag += ort_vec

    k = 1 / n_slots
    const = torch.FloatTensor([k * i for i in range(n_slots)])
    x_diag += const
    x_diag = x_diag % 1
    x_scaled = cfg["x"].min + (cfg["x"].max - cfg["x"].min) * x_diag[1]

    return x_scaled


def split_ood_dataset(ood_dataset):
    """
    Split OOD dataset into two ID datasets with identical slots, but x-coord of the first slot is generated as
    complementary ID.

    """
    z = ood_dataset.latents

    assert z.shape[1] == 2, "Method works only for two objects!"

    # reconvering constant latents
    new_z = torch.zeros((z.shape[0], z.shape[1], z.shape[2] + 3))
    new_z[:, :, :4] = z[:, :, :4]
    new_z[:, :, 5] = z[:, :, -1]
    new_z[:, :, 6:] = 1

    z_left_slot = torch.zeros_like(new_z)
    z_right_slot = torch.zeros_like(new_z)

    z_left_slot[:] = new_z[:, 0, ...].unsqueeze(1)
    z_right_slot[:] = new_z[:, 1, ...].unsqueeze(1)

    for i in range(len(z_left_slot)):
        z_left_slot[i][0][0] = generate_complementary_x_coordinate(z_left_slot[i][0][0])
        z_right_slot[i][0][0] = generate_complementary_x_coordinate(
            z_right_slot[i][0][0]
        )

    return z_left_slot, z_right_slot


def predict_on_id_set(id_dataloader, model):
    latents = []
    figures = []
    with torch.no_grad():
        for images, true_latents in id_dataloader:
            true_figures = images[:, :-1, ...]
            images = images[:, -1, ...].squeeze(1)

            output = model(images)
            predicted_figures = output["reconstructed_figures"]

            figures_reshaped = true_figures.view(
                true_figures.shape[0], true_figures.shape[1], -1
            )

            predicted_figures = predicted_figures.permute(1, 0, 2, 3, 4)
            predicted_figures_reshaped = predicted_figures.reshape(
                predicted_figures.shape[0], predicted_figures.shape[1], -1
            )

            _, indexes = hungarian_slots_loss(
                figures_reshaped, predicted_figures_reshaped
            )

            indexes = torch.LongTensor(indexes)
            predicted_latents = output["predicted_latents"].detach().cpu()
            true_latents = true_latents.detach().cpu()

            # shuffling predicted latents to match true latents
            predicted_latents = predicted_latents.gather(
                1,
                indexes[:, :, 1]
                .unsqueeze(-1)
                .expand(-1, -1, predicted_latents.shape[-1]),
            )
            predicted_figures_reshaped = predicted_figures_reshaped.gather(
                1,
                indexes[:, :, 1]
                .unsqueeze(-1)
                .expand(-1, -1, predicted_figures_reshaped.shape[-1]),
            )
            figures.append(predicted_figures_reshaped.reshape(*predicted_figures.shape))
            latents.append(predicted_latents)

    latents = torch.cat(latents)
    figures = torch.cat(figures)
    return latents, figures


def calculate_optimality(ood_latents, ood_dataloader, model, decoder_hook):
    mse_model = []
    r2_model = []

    mse_decoder = []
    r2_decoder = []

    batch_size_accum = 0
    with torch.no_grad():
        for ood_images, _ in ood_dataloader:
            true_figures = ood_images[:, :-1, ...]
            ood_images = ood_images[:, -1, ...].squeeze(1)

            decoder_output = decoder_hook(
                ood_latents[batch_size_accum : batch_size_accum + len(true_figures)]
            )
            model_output = model(ood_images)

            decoder_recon = decoder_output[0]
            model_recon = model_output["reconstructed_image"]

            # compare generated permuted images with imagined images

            mse_decoder.append(((ood_images - decoder_recon) ** 2).sum(dim=(1, 2, 3)))
            mse_model.append(((ood_images - model_recon) ** 2).sum(dim=(1, 2, 3)))

            r2_decoder.append(image_r2_score(ood_images, decoder_recon) * len(ood_images))
            r2_model.append(image_r2_score(ood_images, model_recon) * len(ood_images))

            batch_size_accum += len(ood_images)
    return (mse_model, r2_model), (mse_decoder, r2_decoder)


def calculate_decoder_optimality(ood_dataset, models, decoder_hooks):
    z_left_slot, z_right_slot = split_ood_dataset(ood_dataset)

    diagonal_dataset_left = data.SpriteWorldDataset(
        len(z_left_slot),
        2,
        cfg,
        sample_mode="skip",
        no_overlap=True,
        delta=0.125,
        transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        z=z_left_slot,
    )

    left_loader = torch.utils.data.DataLoader(
        diagonal_dataset_left,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: utils.collate_fn_normalizer(b, min_offset, scale),
    )

    diagonal_dataset_right = data.SpriteWorldDataset(
        len(z_left_slot),
        2,
        cfg,
        sample_mode="digaonal",
        no_overlap=True,
        delta=0.125,
        transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        z=z_right_slot,
    )

    right_loader = torch.utils.data.DataLoader(
        diagonal_dataset_right,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: utils.collate_fn_normalizer(b, min_offset, scale),
    )

    model_performances = []
    decoder_performances = []
    for (model, decoder_hook) in tqdm.tqdm(zip(models, decoder_hooks)):
        left_latents, left_figures = predict_on_id_set(left_loader, model)
        right_latents, right_figures = predict_on_id_set(right_loader, model)
        ood_latents = torch.cat(
            (left_latents[:, 1, :].unsqueeze(1), right_latents[:, 1, :].unsqueeze(1)),
            dim=1,
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda b: utils.collate_fn_normalizer(b, min_offset, scale),
        )

        model_performance, decoder_performance = calculate_optimality(
            ood_latents, ood_loader, model, decoder_hook
        )
        model_performances.append(model_performance)
        decoder_performances.append(decoder_performance)
    return model_performances, decoder_performances
