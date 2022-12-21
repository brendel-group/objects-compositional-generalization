from src import config
from src import data
from src import models


def test_SlotMLPAdditive():
    cfg = config.SpriteWorldConfig()

    n_slots = 2
    img_h = 64
    img_w = 64
    in_channels = 3
    n_slot_latents = 8

    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=100,
        n_slots=n_slots,
        no_overlap=True,
        sample_mode="diagonal",
        delta=0.125,
        img_h=img_h,
        img_w=img_w,
    )
    model = models.SlotMLPAdditive(
        in_channels=in_channels, n_slots=n_slots, n_slot_latents=n_slot_latents
    )
    output = model(dataset[0][0])

    assert len(output) == 3
    assert output[0].shape == (1, in_channels, img_h, img_w)
    assert output[1].shape == (1, n_slots, n_slot_latents)
    assert len(output[2]) == n_slots


def test_SlotMLPMonolithic():
    cfg = config.SpriteWorldConfig()

    n_slots = 2
    img_h = 64
    img_w = 64
    in_channels = 3
    n_slot_latents = 8

    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=100,
        n_slots=n_slots,
        no_overlap=True,
        sample_mode="diagonal",
        delta=0.125,
        img_h=img_h,
        img_w=img_w,
    )
    model = models.SlotMLPMonolithic(
        in_channels=in_channels, n_slots=n_slots, n_slot_latents=n_slot_latents
    )
    output = model(dataset[0][0])

    assert len(output) == 2
    assert output[0].shape == (1, in_channels, img_h, img_w)
    assert output[1].shape == (1, n_slots, n_slot_latents)


def test_SlotMLPEncoder():
    cfg = config.SpriteWorldConfig()

    n_slots = 2
    img_h = 64
    img_w = 64
    in_channels = 3
    n_slot_latents = 8

    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=100,
        n_slots=n_slots,
        no_overlap=True,
        sample_mode="diagonal",
        delta=0.125,
        img_h=img_h,
        img_w=img_w,
    )
    model = models.SlotMLPEncoder(
        in_channels=in_channels, n_slots=n_slots, n_slot_latents=n_slot_latents
    )
    output = model(dataset[0][0])

    assert output.shape == (1, n_slots, n_slot_latents)


def test_SlotMLPAdditiveDecoder():
    cfg = config.SpriteWorldConfig()

    n_slots = 2
    img_h = 64
    img_w = 64
    in_channels = 3
    n_slot_latents = 8

    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=100,
        n_slots=n_slots,
        no_overlap=True,
        sample_mode="diagonal",
        delta=0.125,
        img_h=img_h,
        img_w=img_w,
    )
    model = models.SlotMLPAdditiveDecoder(
        in_channels=in_channels, n_slots=n_slots, n_slot_latents=n_slot_latents
    )
    output = model(dataset[0][1])

    assert len(output) == 2
    assert output[0].shape == (1, in_channels, img_h, img_w)
    assert len(output[1]) == n_slots


def test_SlotMLPMonolithicDecoder():
    cfg = config.SpriteWorldConfig()

    n_slots = 2
    img_h = 64
    img_w = 64
    in_channels = 3
    n_slot_latents = 8

    dataset = data.SpriteWorldDataset(
        cfg=cfg,
        n_samples=100,
        n_slots=n_slots,
        no_overlap=True,
        sample_mode="diagonal",
        delta=0.125,
        img_h=img_h,
        img_w=img_w,
    )
    model = models.SlotMLPMonolithicDecoder(
        in_channels=in_channels, n_slots=n_slots, n_slot_latents=n_slot_latents
    )
    output = model(dataset[0][1])

    assert output.shape == (1, in_channels, img_h, img_w)
