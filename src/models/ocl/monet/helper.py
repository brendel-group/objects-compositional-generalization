import hydra
import omegaconf
import os


def get_monet_model(num_slots, latent_size, device):
    # relative path to the config file
    path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        "monet.yaml",
    )
    config = omegaconf.OmegaConf.load(path)
    config.model.latent_size = latent_size
    config.model.encoder_params.mlp_output_size = latent_size * 2
    config.model.decoder_params.input_channels = latent_size + 2
    config.model.num_slots = num_slots

    model = hydra.utils.instantiate(config.model).to(device)
    return model
