import argparse
import os

import src
import src.metrics as metrics
import torch
from src.datasets import utils as data_utils
from src.models import base_models, slot_attention
from src.utils import training_utils
from torch.func import jacfwd


def load_model_and_hook(path, model_name, softmax=True, sampling=True):
    # Load the checkpoint
    checkpoint = torch.load(path)

    # Determine which model to load based on the model name
    if model_name == "SlotAttention":
        encoder = slot_attention.SlotAttentionEncoder(
            resolution=(64, 64),
            hid_dim=16,
            ch_dim=32,
            dataset_name="dsprites",
        )
        decoder = slot_attention.SlotAttentionDecoder(
            hid_dim=16,
            ch_dim=32,
            resolution=(64, 64),
            dataset_name="dsprites",
        )
        model = slot_attention.SlotAttentionAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            num_slots=2,
            num_iterations=3,
            hid_dim=16,
            dataset_name="dsprites",
            sampling=sampling,
            softmax=softmax,
        )
        decoder_hook = model.decode
    elif model_name == "SlotMLPAdditive":
        model = base_models.SlotMLPAdditive(3, 2, 16)
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


def calculate_contrast(out, decoder_hook):
    """
    Calculate the contrast score
    """
    latents = out["predicted_latents"]

    jac = jacfwd(decoder_hook)(latents)
    jac_right = jac[0].flatten(1, 4).flatten(2, 3)  # taking the reconstruction jacobian

    (_, _, weighted_comp) = metrics.compositional_contrast(jac_right, 16)

    return weighted_comp.detach().cpu().numpy()


def calculate_identifiability(id_loader, ood_loader, model):
    """
    Calculate the identifiability score
    """
    id_score_id, id_score_ood = metrics.identifiability_score(
        model,
        id_loader,
        ood_loader,
        [2],
        "cuda",
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


def calculate_encoder_consistency(out):
    """
    Calculate the encoder consistency score
    """
    consistency_encoder_loss, _ = metrics.hungarian_slots_loss(
        out["sampled_latents"],
        out["predicted_sampled_latents"],
        "cuda",
    )
    return consistency_encoder_loss.detach().cpu().numpy()


def calculate_ari(images, out):
    true_figures = images[:, :-1, ...].cuda()
    images = images[:, -1, ...].cuda()

    true_masks = training_utils.get_masks(images, true_figures)
    ari_score = metrics.ari(
        true_masks,
        out["reconstructed_masks"].detach().permute(1, 0, 2, 3, 4),
    )
    return ari_score.detach().cpu().numpy()


def evaluate(
    data_path,
    n_slots,
    mixed,
    model_name,
    model_path,
    softmax,
    sampling,
    sample_mode_test_id,
    sample_mode_test_ood,
    batch_size,
):
    data_path = os.path.join(data_utils.data_path, "dsprites")

    wrapper = src.datasets.wrappers.get_wrapper(
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

    # example of how you could load multiple models, feel free to change this
    paths = "MODEL PATH"
    model_name = "SlotMLPAdditive"
    softmax = False  # only for SlotAttention
    sampling = False  # only for SlotAttention
    paths_and_names = []
    for name in os.listdir(paths):
        if name.endswith(".pt"):
            paths_and_names.append((os.path.join(paths, name), model_name))

    models = []
    hooks = []
    for path, name in paths_and_names:
        model, decoder_hook = load_model_and_hook(
            path, name, softmax=softmax, samplig=sampling
        )
        models.append(model)
        hooks.append(decoder_hook)

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
            id_loader, ood_loader, model
        )

        id_r2, ood_r2 = 0, 0
        id_mse, ood_mse = 0, 0
        id_ari_score, ood_ari_score = 0, 0
        id_consistency, ood_consistency = 0, 0
        id_contrast, ood_contrast = 0, 0

        for i, (id_batch, ood_batch) in enumerate(zip(id_loader, ood_loader)):
            id_images, _ = id_batch
            id_images = id_images[:, -1, ...].cuda()  # taking the last image

            ood_images, _ = ood_batch
            ood_images = ood_images[:, -1, ...].cuda()  # taking the last image

            id_out = model(id_images)
            ood_out = model(ood_images)

            id_contrast += calculate_contrast(id_out, hook)
            ood_contrast += calculate_contrast(ood_out, hook)

            id_r2 += calculate_image_r2(id_images, id_out)
            ood_r2 += calculate_image_r2(ood_images, ood_out)

            id_consistency += calculate_encoder_consistency(id_out).mean()
            ood_consistency += calculate_encoder_consistency(ood_out).mean()

            id_mse += calculate_image_mse(id_images, id_out)
            ood_mse += calculate_image_mse(ood_images, ood_out)

            id_ari_score += calculate_ari(id_images, id_out)
            ood_ari_score += calculate_ari(ood_images, ood_out)

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
        "--data_path",
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
        help="Path to the model folder.",
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

    args = parser.parse_args()
    evaluate(**vars(args))
