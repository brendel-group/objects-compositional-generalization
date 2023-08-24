from src.models import base_models, slot_attention
from src.datasets import utils as data_utils
import src.metrics as metrics

import os
import src
import torch

# from functorch import jacfwd
from torch.func import jacfwd

import numpy as np
import pickle


def load_model_and_hook(path, model_name):
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
            sampling=True,
            softmax=True,
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


def compositional_contrast(jac, slot_dim):
    batch_size, obs_dim, z_dim = jac.shape[0], jac.shape[1], jac.shape[2]
    num_slots = int(z_dim / slot_dim)
    jac = jac.reshape(
        batch_size * obs_dim, z_dim
    )  # batch_size*obs_dim x num_slots*slot_dim
    slot_rows = torch.stack(
        torch.split(jac, slot_dim, dim=1)
    )  # num_slots x batch_size*obs_dim x slot_dim
    slot_norms = torch.norm(slot_rows, dim=2)  # num_slots x batch_size*obs_dim
    slot_norms = slot_norms.view(num_slots, batch_size, obs_dim).permute(
        1, 0, 2
    )  # batch_size x num_slots x obs_dim
    slot_norms += 1e-12
    slot_norms_max = slot_norms.sum(1) / num_slots
    slot_norms_norm = slot_norms / slot_norms_max.unsqueeze(1).repeat(1, num_slots, 1)
    max_norm_all = torch.max(slot_norms_max, 1)[0]
    weights = slot_norms_max / max_norm_all.unsqueeze(1).repeat(1, obs_dim)

    comp_conts = 0
    comp_conts_norm = 0
    for i in range(num_slots):
        for j in range(i, num_slots - 1):
            comp_conts += slot_norms[:, i] * slot_norms[:, j + 1]
            comp_conts_norm += slot_norms_norm[:, i] * slot_norms_norm[:, j + 1]
    comp_cont = (comp_conts).sum(1).mean()
    comp_cont_norm = (comp_conts_norm).sum(1).mean()
    weight_comp_cont_norm = ((comp_conts_norm) * weights).sum(1)

    return comp_cont, comp_cont_norm, weight_comp_cont_norm


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

    (_, _, weighted_comp) = compositional_contrast(jac_right, 16)

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


def evaluate():
    data_path = os.path.join(data_utils.data_path, "dsprites")
    n_samples_test = 5000
    n_slots = 2
    mixed = False
    delta = 0.125
    no_overlap = True
    batch_size = 5
    sample_mode_test_id = "diagonal"
    sample_mode_test_ood = "no_overlap_off_diagonal"

    wrapper = src.datasets.wrappers.get_wrapper(
        "dsprites",
        path=data_path,
        load=True,
        save=False,
    )

    id_loader = wrapper.get_test_loader(
        n_samples_test,
        n_slots,
        sample_mode_test_id,
        delta,
        no_overlap,
        batch_size,
        mixed=mixed,
    )

    ood_loader = wrapper.get_test_loader(
        n_samples_test,
        n_slots,
        sample_mode_test_ood,
        delta,
        no_overlap,
        batch_size,
        mixed=mixed,
    )

    # example of how you could load multiple models, feel free to change this
    paths = "/mnt/qb/work/bethge/apanfilov27/slurm_scripts/models/SlotMLPAdditive_2obj_300_enccons_no_overlaps"
    paths_and_names = []
    for name in os.listdir(paths):
        if name.endswith(".pt"):
            paths_and_names.append((os.path.join(paths, name), "SlotMLPAdditive"))

    models = []
    hooks = []
    for path, name in paths_and_names:
        model, decoder_hook = load_model_and_hook(path, name)
        models.append(model)
        hooks.append(decoder_hook)

    cast_models_to_cuda(models)

    id_id_scores = []
    ood_id_scores = []
    id_contrasts = []
    ood_contrasts = []
    id_image_r2 = []
    ood_image_r2 = []
    id_encoder_consistency = []
    ood_encoder_consistency = []


    # evaluating provided models
    for model, hook in zip(models, hooks):
        # mean id scores
        id_id_score, ood_id_score = calculate_identifiability(
            id_loader, ood_loader, model
        )
        id_id_score, ood_id_score = 0, 0

        id_r2, ood_r2 = 0, 0
        id_consistency, ood_consistency = 0, 0
        id_contrast, ood_contrast = 0, 0

        for i, (id_batch, ood_batch) in enumerate(zip(id_loader, ood_loader)):
            id_images, _ = id_batch
            id_images = id_images[:, -1, ...].cuda() # taking the last image


            ood_images, _ = ood_batch
            ood_images = ood_images[:, -1, ...].cuda() # taking the last imag

            id_out = model(id_images)
            ood_out = model(ood_images)

            id_contrast += calculate_contrast(id_out, hook)
            ood_contrast += calculate_contrast(ood_out, hook)

            id_r2 += calculate_image_r2(id_images, id_out)
            ood_r2 += calculate_image_r2(ood_images, ood_out)

            id_consistency += calculate_encoder_consistency(id_out).mean()
            ood_consistency += calculate_encoder_consistency(ood_out).mean()

        id_contrasts.append(id_contrast * batch_size / n_samples_test)
        ood_contrasts.append(ood_contrast * batch_size / n_samples_test)

        id_image_r2.append(id_r2 * batch_size / n_samples_test)
        ood_image_r2.append(ood_r2 * batch_size / n_samples_test)

        id_encoder_consistency.append(id_consistency * batch_size / n_samples_test)
        ood_encoder_consistency.append(ood_consistency * batch_size / n_samples_test)

        id_id_scores.append(id_id_score)
        ood_id_scores.append(ood_id_score)


    print("id_id_scores", id_id_scores)
    print("ood_id_scores", ood_id_scores)
    print("id_contrasts", id_contrasts)
    print("ood_contrasts", ood_contrasts)
    print("id_image_r2", id_image_r2)
    print("ood_image_r2", ood_image_r2)

if __name__ == "__main__":
    evaluate()
