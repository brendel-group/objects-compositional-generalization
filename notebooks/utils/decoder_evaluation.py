import sys

# print(sys.path)
sys.path.append("D:\\git_projects\\bethgelab\\lab_rotation\\object_centric_ood")
# print(sys.path)

import numpy as np
import torch
import random

from src.datasets import data, utils, configs
from src.metrics import hungarian_slots_loss


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
    """
    Generate x-coordinate of the first slot in the support ID dataset.

    Parameters:
    x: float, x-coordinate of ood-object
    delta: float, maximum deviation of the generated x-coordinate from x (0.125 matches original dataset)
    n_slots: int, number of objects or slots in the dataset

    Returns:
    x_scaled: float, x-coordinate of the first slot in the ID dataset
    """

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
    Split OOD dataset into two ID datasets with identical slots, but x-coordinate of the first slot is generated as
    complementary ID.

    Parameters:
    ood_dataset: Dataset, the out-of-distribution dataset

    Returns:
    z_left_slot, z_right_slot: Tensors, the split in-distribution datasets
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

    z_left_slot[:] = new_z[:, 0, ...].unsqueeze(1).clone()
    z_right_slot[:] = new_z[:, 1, ...].unsqueeze(1).clone()

    for i in range(len(z_left_slot)):
        z_left_slot[i][0][0] = generate_complementary_x_coordinate(z_left_slot[i][0][0])
        z_right_slot[i][0][0] = generate_complementary_x_coordinate(
            z_right_slot[i][0][0]
        )

    return z_left_slot, z_right_slot


def predict_on_id_set(id_dataloader, model, device="cpu"):
    """
    Given an ID dataloader, it makes predictions using the given model and returns the latents and figures.

    Parameters:
    id_dataloader: DataLoader, dataloader for the in-distribution dataset
    model: trained model
    device: str, device to run the model on, default is "cpu"

    Returns:
    latents, figures: Tensors, the predicted latents and figures
    """
    latents = []
    figures = []
    with torch.no_grad():
        for images, true_latents in id_dataloader:
            true_figures = images[:, :-1, ...]
            images = images[:, -1, ...].squeeze(1)

            output = model(images)
            predicted_figures = output["reconstructed_figures"]

            true_figures_reshaped = true_figures.view(
                true_figures.shape[0], true_figures.shape[1], -1
            )

            predicted_figures = predicted_figures.permute(1, 0, 2, 3, 4).detach()
            predicted_figures_reshaped = predicted_figures.reshape(
                predicted_figures.shape[0], predicted_figures.shape[1], -1
            )

            _, indexes = hungarian_slots_loss(
                true_figures_reshaped, predicted_figures_reshaped, device=device
            )

            indexes = torch.LongTensor(indexes).detach().to(device)
            predicted_latents = output["predicted_latents"].detach()

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
            figures.append(
                predicted_figures_reshaped.cpu().detach().reshape(*true_figures.shape)
            )
            latents.append(predicted_latents.detach())

    latents = torch.cat(latents)
    figures = torch.cat(figures)
    return latents, figures


def calculate_optimality(ood_latents, ood_figures, ood_dataloader, model, decoder_hook):
    """
    Calculates and returns the mean squared error (MSE) between the OOD images and the images reconstructed by the model
    and the decoder from provided latents.

    Parameters:
    ood_latents: Tensor, the OOD latents (generated earlier)
    ood_figures: Tensor, the OOD figures (generated earlier)
    ood_dataloader: DataLoader, dataloader for the out-of-distribution dataset
    model: trained model to make predictions
    decoder_hook: decoder hook of the model, to reconstruct the images using the decoder

    Returns:
    dict: contains the MSE for the model, the decoder, and the figures
    """

    mse_model = []
    figures_mse_model = []

    mse_decoder = []

    reconstructed_images = []

    batch_size_accum = 0
    with torch.no_grad():
        for ood_images, _ in ood_dataloader:
            ood_images = ood_images[:, -1, ...].squeeze(1)

            decoder_output = decoder_hook(
                ood_latents[batch_size_accum : batch_size_accum + len(ood_images)]
            )
            model_output = model(ood_images)

            decoder_recon = decoder_output[0].cpu().detach()
            model_recon = model_output["reconstructed_image"].cpu().detach()
            ood_images = ood_images.cpu().detach()
            # compare generated permuted images with imagined images

            mse_decoder.append((ood_images - decoder_recon).square().sum(dim=(1, 2, 3)))
            mse_model.append((ood_images - model_recon).square().sum(dim=(1, 2, 3)))
            figures_mse_model.append(
                (
                    ood_figures[batch_size_accum : batch_size_accum + len(ood_images)]
                    .cpu()
                    .detach()
                    - ood_images
                )
                .square()
                .sum(dim=(1, 2, 3))
            )
            reconstructed_images.append(model_recon)
            batch_size_accum += len(ood_images)
    return {
        "mse_model": torch.cat(mse_model),
        "mse_decoder": torch.cat(mse_decoder),
        "figures_mse_model": torch.cat(figures_mse_model),
        "reconstructed_images": torch.cat(reconstructed_images),
    }


def get_left_right_datasets(ood_dataset):
    """
    Given an OOD dataset, it splits the dataset into two datasets, one for the left slot and one for the right slot.

    Parameters:
    ood_dataset: Dataset, the out-of-distribution dataset

    Returns:
    left_slot_dataset, right_slot_dataset: Dataset, the left and right slot datasets

    """
    z_left_slot, z_right_slot = split_ood_dataset(
        ood_dataset
    )  # second slots of these datasets are original objects

    ##### Getting support images #####
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

    diagonal_dataset_right = data.SpriteWorldDataset(
        len(z_right_slot),
        2,
        cfg,
        sample_mode="skip",
        no_overlap=True,
        delta=0.125,
        transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]),
        z=z_right_slot,
    )
    return diagonal_dataset_left, diagonal_dataset_right


def calculate_decoder_optimality(
    ood_dataset,
    diagonal_dataset_left,
    diagonal_dataset_right,
    models,
    decoder_hooks,
    device="cpu",
    output_figures=False,
):
    """
    Calculates the optimality of the decoder using a given OOD dataset, a list of models, decoder hooks.

    Parameters:
    ood_dataset: Dataset, the out-of-distribution dataset
    diagonal_dataset_left: Dataset, the left slot dataset
    diagonal_dataset_right: Dataset, the right slot dataset
    models: list of trained models to make predictions
    decoder_hooks: list of functions, to reconstruct the images using the decoder hooks
    device: str, device to run the models on, default is "cpu"

    Returns:
    dict: contains the performances of models, matched latents and figures, and original latents and figures
    """

    left_loader = torch.utils.data.DataLoader(
        diagonal_dataset_left,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: utils.collate_fn_normalizer(
            b, min_offset, scale, device=device
        ),
    )

    right_loader = torch.utils.data.DataLoader(
        diagonal_dataset_right,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: utils.collate_fn_normalizer(
            b, min_offset, scale, device=device
        ),
    )

    model_performances = []
    ood_matched_latents = []
    ood_matched_figures = []
    left_latents_list = []
    right_latents_list = []
    left_figures_list = []
    right_figures_list = []
    reconstructed_images_list = []

    for model, decoder_hook in tqdm.tqdm(zip(models, decoder_hooks)):
        left_latents, left_figures = predict_on_id_set(left_loader, model, device)
        right_latents, right_figures = predict_on_id_set(right_loader, model, device)

        # collecting latents for ood_dataset i.e., second slot of each pair
        ood_latents = torch.cat(
            (left_latents[:, 1, :].unsqueeze(1), right_latents[:, 1, :].unsqueeze(1)),
            dim=1,
        )
        # collecting figures for ood_dataset i.e., second figure of each pair
        ood_figures = torch.cat(
            (
                left_figures[:, 1, ...].unsqueeze(1),
                right_figures[:, 1, ...].unsqueeze(1),
            ),
            dim=1,
        ).sum(dim=1)

        ood_loader = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda b: utils.collate_fn_normalizer(
                b, min_offset, scale, device=device
            ),
        )

        # calculating optimality
        # predicting with decoder on ood_latents & predicting with model on ood_images
        # als o calculating optimality given the figures (should be exactly the same for non-masked models)
        performance = calculate_optimality(
            ood_latents, ood_figures, ood_loader, model, decoder_hook
        )
        if "reconstructed_images" in performance:
            reconstructed_images = performance.pop("reconstructed_images")
            reconstructed_images_list.append(reconstructed_images)

        # collecting results for each model
        model_performances.append(performance)
        ood_matched_latents.append(ood_latents)
        ood_matched_figures.append(ood_figures)
        left_latents_list.append(left_latents)
        right_latents_list.append(right_latents)
        left_figures_list.append(left_figures)
        right_figures_list.append(right_figures)

    output = {
        "model_performances": model_performances,
    }

    if output_figures:
        output["ood_matched_figures"] = ood_matched_figures
        output["reconstructed_images"] = reconstructed_images_list

    return output
    # return {
    # "model_performances": model_performances,
    # "ood_matched_latents": ood_matched_latents,
    # "ood_matched_figures": ood_matched_figures,
    # "reconstructed_images": reconstructed_images_list,
    # "left_slot_output": {
    #     "left_latents": left_latents_list,
    #     "left_figures": left_figures_list,
    # },
    # "right_slot_output": {
    #     "right_latents": right_latents_list,
    #     "right_figures": right_figures_list,
    # },
    # }
