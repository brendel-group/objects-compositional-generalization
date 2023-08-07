from src.models import base_models, slot_attention
from src.datasets import utils as data_utils

import os
import src
import torch

# from functorch import jacfwd
from torch.func import jacfwd
from tqdm import tqdm

import numpy as np


import functorch
print(functorch.__version__)

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
    jac = jac.reshape(batch_size * obs_dim, z_dim)  # batch_size*obs_dim x num_slots*slot_dim
    slot_rows = torch.stack(torch.split(jac, slot_dim, dim=1))  # num_slots x batch_size*obs_dim x slot_dim
    slot_norms = torch.norm(slot_rows, dim=2)  # num_slots x batch_size*obs_dim
    slot_norms = slot_norms.view(num_slots, batch_size, obs_dim).permute(1, 0, 2)  # batch_size x num_slots x obs_dim
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
    weight_comp_cont_norm = ((comp_conts_norm) * weights).sum(1).mean()

    return comp_cont, comp_cont_norm, weight_comp_cont_norm


def cast_models_to_cuda(models):
    for model in models:
        model.cuda()

def comp_eval(id_loader, ood_loader, models, hooks):
    def process_loader(loader, model, decoder_hook):
        accum_comp_contrast = 0
        for (images, latents) in tqdm(loader):
            images = images[:, -1, ...].to("cuda")
            out = model(images)
            latents = out["predicted_latents"]

            jac = jacfwd(decoder_hook)(latents)
            jac_right = jac[0].flatten(1, 4).flatten(2, 3) # taking the reconstruction jacobian

            (_, _, weighted_comp) = compositional_contrast(jac_right, 16)

            accum_comp_contrast += weighted_comp.detach().cpu().numpy()
        return accum_comp_contrast

    contrasts = []
    for model, decoder_hook in zip(models, hooks):
        accum_comp_contrast_id = process_loader(id_loader, model, decoder_hook)
        accum_comp_contrast_ood = process_loader(ood_loader, model, decoder_hook)
        contrasts.append((accum_comp_contrast_id, accum_comp_contrast_ood))

    return contrasts

def evaluate():
    data_path = os.path.join(data_utils.data_path, "dsprites")
    n_samples_test = 5000
    n_slots = 4
    mixed = True
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

    # paths = "/mnt/qb/work/bethge/apanfilov27/slurm_scripts/models/SlotAttention_2obj_nooverlaps_nomasks_nosampling_vanilla_600"
    # paths_and_names = []
    # for name in os.listdir(paths):
    #     if name.endswith(".pt"):
    #         paths_and_names.append((os.path.join(paths, name), "SlotAttention"))

    paths = "/mnt/qb/work/bethge/apanfilov27/slurm_scripts/models/SlotAttention_MIXEDobj_vanilla_400"
    paths_and_names = []
    for name in os.listdir(paths):
        if name.endswith(".pt"):
            paths_and_names.append((os.path.join(paths, name), "SlotAttention"))

    # paths = "/mnt/qb/work/bethge/apanfilov27/slurm_scripts/models/SlotAttention_2obj_400_vanilla_no_overlaps"
    # paths_and_names = [
    #     (os.path.join(paths, "SlotAttention_2obj_400_vanilla_no_overlaps_2023.pt"), "SlotAttention"),
    #     (os.path.join(paths, "SlotAttention_2obj_400_vanilla_no_overlaps_2024.pt"), "SlotAttention"),
    #     (os.path.join(paths, "SlotAttention_2obj_400_vanilla_no_overlaps_2025.pt"), "SlotAttention"),
    #     (os.path.join(paths, "SlotAttention_2obj_400_vanilla_no_overlaps_2026.pt"), "SlotAttention"),
    #     (os.path.join(paths, "SlotAttention_2obj_400_vanilla_no_overlaps_2027.pt"), "SlotAttention"),
    # ]

    # paths = "/mnt/qb/work/bethge/apanfilov27/slurm_scripts/models/SlotMLPAdditive_2obj_300_vanilla_no_overlaps"
    # paths_and_names = [
    #     (os.path.join(paths, "SlotMLPAdditive_2obj_300_vanilla_no_overlaps_2023.pt"), "SlotMLPAdditive"),
    #     (os.path.join(paths, "SlotMLPAdditive_2obj_300_vanilla_no_overlaps_2024.pt"), "SlotMLPAdditive"),
    #     (os.path.join(paths, "SlotMLPAdditive_2obj_300_vanilla_no_overlaps_2027.pt"), "SlotMLPAdditive"),
    #     (os.path.join(paths, "SlotMLPAdditive_2obj_300_vanilla_no_overlaps_2028.pt"), "SlotMLPAdditive"),
    #     (os.path.join(paths, "SlotMLPAdditive_2obj_300_vanilla_no_overlaps_2031.pt"), "SlotMLPAdditive"),
    # ]
    models = []
    hooks = []
    for (path, name) in paths_and_names:
        model, decoder_hook = load_model_and_hook(path, name)
        models.append(model)
        hooks.append(decoder_hook)

    cast_models_to_cuda(models)

    contrasts = comp_eval(id_loader, ood_loader, models, hooks)

    id_contrasts = np.array([c[0] for c in contrasts])
    ood_contrasts = np.array([c[1] for c in contrasts])

    id_contrasts = id_contrasts * batch_size / n_samples_test
    ood_contrasts = ood_contrasts * batch_size / n_samples_test

    print("ID contrasts: ", np.mean(id_contrasts), "+-", np.std(id_contrasts))
    print("OOD contrasts: ", np.mean(ood_contrasts), "+-", np.std(ood_contrasts))

if __name__ == "__main__":
    evaluate()
