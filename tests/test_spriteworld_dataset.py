import pytest

from src.datasets import data, configs


@pytest.mark.parametrize(
    "config_dict",
    [
        {},
        {
            "x": configs.Range(0.1, 0.9),
            "y": configs.Range(0.1, 0.9),
            "shape": ["circle"],
            "angle": configs.Range(0, 90),
        },
    ],
)
def test_cfg(config_dict):
    custom_cfg = config.SpriteWorldConfig(**config_dict)

    assert custom_cfg.get_total_latent_dim == 8
    assert len(custom_cfg.get_latents_metadata()) == 8
    assert len(custom_cfg.get_ranges()) == 8

    assert custom_cfg.x.min == config_dict.get("x", config.Range(0.1, 0.9)).min
    assert custom_cfg.x.max == config_dict.get("x", config.Range(0.1, 0.9)).max
    assert custom_cfg.y.min == config_dict.get("y", config.Range(0.2, 0.8)).min
    assert custom_cfg.y.max == config_dict.get("y", config.Range(0.2, 0.8)).max
    assert custom_cfg.shape == config_dict.get(
        "shape",
        [
            "triangle",
            "square",
        ],
    )

    assert custom_cfg.get_ranges()["x"].min == custom_cfg.x.min
    assert custom_cfg.get_ranges()["x"].max == custom_cfg.x.max
    assert custom_cfg.get_ranges()["shape"].min == 0
    assert custom_cfg.get_ranges()["shape"].max == len(custom_cfg.shape) - 1


@pytest.mark.timeout(60)
@pytest.mark.parametrize("n_samples", [1, 10, 1000])
@pytest.mark.parametrize("n_slots", [2, 3])
@pytest.mark.parametrize("no_overlap", [True, False])
@pytest.mark.parametrize("delta", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("sample_mode", ["diagonal", "off_diagonal"])
def test_diagonal(n_samples, n_slots, no_overlap, delta, sample_mode):
    cfg = config.SpriteWorldConfig()
    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=n_samples,
        n_slots=n_slots,
        no_overlap=no_overlap,
        sample_mode=sample_mode,
        delta=delta,
    )

    assert len(dataset) == n_samples


@pytest.mark.timeout(60)
@pytest.mark.parametrize("n_samples", [1, 10, 1000])
@pytest.mark.parametrize("n_slots", [1, 2, 3])
def test_random(n_samples, n_slots):
    cfg = config.SpriteWorldConfig()
    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=n_samples,
        n_slots=n_slots,
        no_overlap=True,
        sample_mode="random",
    )

    assert len(dataset) == n_samples
    assert dataset.delta == 1
