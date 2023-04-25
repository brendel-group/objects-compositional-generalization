import hydra
import omegaconf
import os


def get_space_model(num_slots, latent_size, device):
    # relative path to the config file
    path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        "space.yaml",
    )
    config = omegaconf.OmegaConf.load(path)
    config.model.fg_params.G = num_slots
    config.model.bg_params.K = 1
    config.model.fg_params.z_what_dim = latent_size

    config.model.num_slots = num_slots * num_slots + 1

    model = hydra.utils.instantiate(config.model).to(device)
    return model