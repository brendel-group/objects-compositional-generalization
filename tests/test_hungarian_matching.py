import torch
from src.metrics import hungarian_slots_loss


def test_hungarian_slots_loss_matching():
    # Define constants
    batch_size, n_slots, n_latents = 10, 4, 5
    device = 'cpu'

    # Generate predictable tensors
    true_latents = torch.zeros((batch_size, n_slots, n_latents))
    for i in range(n_slots):
        true_latents[:, i, :] = i

    # Create a shuffled version
    predicted_latents = true_latents.clone()
    for i in range(batch_size):
        predicted_latents[i] = predicted_latents[i, torch.randperm(n_slots)]

    # Move tensors to the specified device
    true_latents = true_latents.to(device)
    predicted_latents = predicted_latents.to(device)

    # Compute the loss and get the transposed indices
    loss, transposed_indices = hungarian_slots_loss(true_latents, predicted_latents, device)

    # Check that the loss is zero
    assert torch.isclose(loss, torch.tensor(0., device=device), atol=1e-6)

    # Check that the slots are correctly matched
    for i in range(batch_size):
        # Check the second column of the sorted indices, should result in a tensor of [0, 1, 2, ..., n_slots-1]
        assert torch.equal(torch.sort(transposed_indices[i], dim=0)[0][:, 1], torch.arange(n_slots))
