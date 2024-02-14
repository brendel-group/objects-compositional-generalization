import argparse
import os

import torch
from torch.func import jacfwd

import src.metrics as metrics
from src.datasets import wrappers
from src.models import base_models, slot_attention
from src.utils import training_utils


def load_model_and_hook(
    path, model_name, num_slots, n_slot_latents, softmax=True, sampling=True
):
    # Load the checkpoint
    checkpoint = torch.load(path)

    # Determine which model to load based on the model name
    if model_name == "SlotAttention":
        encoder = slot_attention.SlotAttentionEncoder(
            resolution=(64, 64),
            hid_dim=n_slot_latents,
            ch_dim=32,
        )
        decoder = slot_attention.SlotAttentionDecoder(
            hid_dim=n_slot_latents,
            ch_dim=32,
            resolution=(64, 64),
        )
        model = slot_attention.SlotAttentionAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            num_slots=num_slots,
            num_iterations=3,
            hid_dim=n_slot_latents,
            sampling=sampling,
            softmax=softmax,
        )
        decoder_hook = model.decode
    elif model_name == "SlotMLPAdditive":
        model = base_models.SlotMLPAdditive(3, num_slots, n_slot_latents)
        decoder_hook = model.decoder
    else:
        raise ValueError("Invalid model name")

    # Load the model weights and set the model to eval model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, decoder_hook


def cast_models_to_cuda(models):
    for model in models:
        model.cuda()


def calculate_contrast(out, decoder_hook, n_slot_latents):
    """
    Calculate the contrast score
    """
    latents = out["predicted_latents"]

    jac = jacfwd(decoder_hook)(latents)
    jac_right = jac[0].flatten(1, 4).flatten(2, 3)  # taking the reconstruction jacobian

    (_, _, weighted_comp) = metrics.compositional_contrast(jac_right, n_slot_latents)

    return weighted_comp.detach().cpu().numpy()


def calculate_identifiability(id_loader, ood_loader, model, device):
    """
    Calculate the identifiability score
    """
    id_score_id, id_score_ood = metrics.identifiability_score(
        model,
        id_loader,
        ood_loader,
        [2],
        device,
    )
    return id_score_id, id_score_ood


def calculate_image_r2(images, out):
    """
    Calculate the image r2 score
    """
    x_hat = out["reconstructed_image"]

    r2 = metrics.image_r2_score(true_images=images, predicted_images=x_hat)
    return r2.detach().cpu().numpy()


def calculate_image_mse(images, out):
    """
    Calculate the image mse score
    """
    x_hat = out["reconstructed_image"]

    mse = metrics.reconstruction_loss(images, x_hat)
    return mse.detach().cpu().numpy()


def calculate_encoder_consistency(out, device):
    """
    Calculate the encoder consistency score
    """
    consistency_encoder_loss, _ = metrics.hungarian_slots_loss(
        out["sampled_latents"],
        out["predicted_sampled_latents"],
        device,
    )
    return consistency_encoder_loss.detach().cpu().numpy()


def calculate_ari(images, out, device):
    """
    Calculate the ari score
    """

    true_figures = images[:, :-1, ...].to(device)
    images = images[:, -1, ...].to(device)
    true_masks = training_utils.get_masks(images, true_figures)
    ari_score = metrics.ari(
        true_masks,
        out["reconstructed_masks"].detach().permute(1, 0, 2, 3, 4),
    )
    return ari_score.detach().cpu().numpy()


def evaluate(
    dataset_path,
    n_slots,
    mixed,
    model_name,
    model_path,
    softmax,
    sampling,
    sample_mode_test_id,
    sample_mode_test_ood,
    batch_size,
    num_slots,
    n_slot_latents,
    device="cuda",
):
    data_path = os.path.join(dataset_path, "dsprites")

    wrapper = wrappers.get_wrapper(
        "dsprites",
        path=data_path,
    )

    id_loader = wrapper.get_test_loader(
        n_slots,
        sample_mode_test_id,
        batch_size,
        mixed=mixed,
    )

    ood_loader = wrapper.get_test_loader(
        n_slots,
        sample_mode_test_ood,
        batch_size,
        mixed=mixed,
    )

    # you can add more models here
    models = []
    hooks = []
    model, decoder_hook = load_model_and_hook(
        model_path,
        model_name,
        num_slots,
        n_slot_latents,
        softmax=softmax,
        sampling=sampling,
    )
    models.append(model)
    hooks.append(decoder_hook)

    if device == "cuda":
        cast_models_to_cuda(models)

    id_id_scores = []
    ood_id_scores = []
    id_contrasts = []
    ood_contrasts = []
    id_image_r2 = []
    ood_image_r2 = []
    id_image_mse = []
    ood_image_mse = []
    id_encoder_consistency = []
    ood_encoder_consistency = []
    id_ari = []
    ood_ari = []

    n_samples_test_id = len(id_loader.dataset)
    n_samples_test_ood = len(ood_loader.dataset)
    # evaluating provided models
    for model, hook in zip(models, hooks):
        # mean id scores
        id_id_score, ood_id_score = calculate_identifiability(
            id_loader, ood_loader, model, device
        )

        id_r2, ood_r2 = 0, 0
        id_mse, ood_mse = 0, 0
        id_ari_score, ood_ari_score = 0, 0
        id_consistency, ood_consistency = 0, 0
        id_contrast, ood_contrast = 0, 0

        for i, (id_batch, ood_batch) in enumerate(zip(id_loader, ood_loader)):
            id_images_figures, _ = id_batch
            id_images = id_images_figures[:, -1, ...].to(
                device
            )  # taking the last image

            ood_images_figures, _ = ood_batch
            ood_images = ood_images_figures[:, -1, ...].to(
                device
            )  # taking the last image

            id_out = model(id_images)
            ood_out = model(ood_images)

            if model_name == "SlotMLPAdditive":
                id_contrast += calculate_contrast(id_out, hook, n_slot_latents)
                ood_contrast += calculate_contrast(ood_out, hook, n_slot_latents)

            id_r2 += calculate_image_r2(id_images, id_out)
            ood_r2 += calculate_image_r2(ood_images, ood_out)

            id_consistency += calculate_encoder_consistency(id_out, device).mean()
            ood_consistency += calculate_encoder_consistency(ood_out, device).mean()

            id_mse += calculate_image_mse(id_images, id_out)
            ood_mse += calculate_image_mse(ood_images, ood_out)

            if model_name == "SlotAttention":
                id_ari_score += calculate_ari(id_images_figures, id_out, device)
                ood_ari_score += calculate_ari(ood_images_figures, ood_out, device)

        id_contrasts.append(id_contrast * batch_size / n_samples_test_id)
        ood_contrasts.append(ood_contrast * batch_size / n_samples_test_ood)

        id_image_r2.append(id_r2 * batch_size / n_samples_test_id)
        ood_image_r2.append(ood_r2 * batch_size / n_samples_test_ood)

        id_encoder_consistency.append(id_consistency * batch_size / n_samples_test_id)
        ood_encoder_consistency.append(
            ood_consistency * batch_size / n_samples_test_ood
        )

        id_image_mse.append(id_mse * batch_size / n_samples_test_id)
        ood_image_mse.append(ood_mse * batch_size / n_samples_test_ood)

        id_ari.append(id_ari_score * batch_size / n_samples_test_id)
        ood_ari.append(ood_ari_score * batch_size / n_samples_test_ood)

        id_id_scores.append(id_id_score)
        ood_id_scores.append(ood_id_score)

    print("id_id_scores", id_id_scores)
    print("ood_id_scores", ood_id_scores)
    print("id_contrasts", id_contrasts)
    print("ood_contrasts", ood_contrasts)
    print("id_image_r2", id_image_r2)
    print("ood_image_r2", ood_image_r2)
    print("id_encoder_consistency", id_encoder_consistency)
    print("ood_encoder_consistency", ood_encoder_consistency)
    print("id_image_mse", id_image_mse)
    print("ood_image_mse", ood_image_mse)
    print("id_ari", id_ari)
    print("ood_ari", ood_ari)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start evaluation.")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data",
        help="Path to the dataset folder.",
    )
    parser.add_argument(
        "--n_slots",
        type=int,
        default=2,
        help="Number of slots.",
    )
    parser.add_argument(
        "--mixed",
        choices=[True, False],
        default=False,
        help="Whether to use mixed dataset.",
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "SlotMLPAdditive",
            "SlotAttention",
        ],
        default="SlotAttention",
        help="Model to use.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models",
        help="Path to the saved checkpoint.",
    )
    parser.add_argument(
        "--softmax",
        choices=[True, False],
        default=True,
        help="Whether to use softmax in SlotAttention.",
    )
    parser.add_argument(
        "--sampling",
        choices=[True, False],
        default=True,
        help="Whether to use sampling in SlotAttention.",
    )
    parser.add_argument(
        "--sample_mode_test_id",
        type=str,
        default="diagonal",
        help="Sample mode for the test set.",
    )

    parser.add_argument(
        "--sample_mode_test_ood",
        type=str,
        default="no_overlap_off_diagonal",
        help="Sample mode for the test set.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--num_slots",
        type=int,
        default=2,
        help="Number of slots.",
    )
    parser.add_argument(
        "--n_slot_latents",
        type=int,
        default=16,
        help="Number of latents.",
    )

    args = parser.parse_args()
    evaluate(**vars(args))
