import hydra
import omegaconf
import os


def get_genesis_model(num_slots, latent_size, device):
    # relative path to the config file
    path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        "genesis.yaml",
    )
    config = omegaconf.OmegaConf.load(path)

    config.model.num_slots = num_slots
    config.model.component_vae_params.latent_size = latent_size
    config.model.component_vae_params.encoder_params.mlp_output_size = latent_size * 2
    config.model.component_vae_params.decoder_params.input_channels = latent_size + 2

    model = hydra.utils.instantiate(config.model).to(device)
    return model
