import pytest
import torch

from src import training_utils


@pytest.mark.parametrize(
    "n_samples",
    [2, 10],
)
@pytest.mark.parametrize(
    "n_slots",
    [1, 2, 3],
)
def test_r2_score(n_samples, n_slots):
    n_latents = n_slots
    true_latents = torch.rand(n_samples, n_slots, n_latents)
    predicted_latents = true_latents.clone()
    indices = torch.tensor([[i] * 2 for i in range(n_slots)] * n_samples)
    indices = indices.reshape(n_samples, n_slots, 2)
    avg_r2_score, r2_score_raw = training_utils.calculate_r2_score(
        true_latents, predicted_latents, indices
    )

    assert avg_r2_score == 1.0
    assert torch.all(r2_score_raw == torch.ones(n_latents * n_slots))
