import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from src.metrics import hungarian_slots_loss


@pytest.mark.parametrize(
    "batch_size,n_slots,n_latents",
    [
        (10, 4, 5),
        (5, 3, 2),
        (2, 5, 4),
        (1, 6, 7),
        # Add more combinations as needed...
    ],
)
def test_hungarian_slots_loss_permutation(batch_size, n_slots, n_latents):
    device = "cpu"

    # Generate predictable tensors
    true_latents = torch.zeros((batch_size, n_slots, n_latents))
    for i in range(n_slots):
        true_latents[:, i, :] = i

    # Create a shuffled version
    predicted_latents = true_latents.clone()
    for i in range(batch_size):
        perm = torch.randperm(n_slots)
        predicted_latents[i] = predicted_latents[i, perm]

        # Save the permutation for each batch for later comparison
        if i == 0:
            perms = perm.unsqueeze(0)
        else:
            perms = torch.cat((perms, perm.unsqueeze(0)), dim=0)

    # Move tensors to the specified device
    true_latents = true_latents.to(device)
    predicted_latents = predicted_latents.to(device)
    perms = perms.to(device)

    # Compute the loss and get the transposed indices
    loss, transposed_indices = hungarian_slots_loss(
        true_latents, predicted_latents, device
    )

    # Check that the loss is zero
    assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)
    assert torch.equal(torch.argsort(perms), transposed_indices.transpose(1, 2)[:, 1])


def jacks_implementation(xs, xhs, n_slots, device):
    # Compute the MSE loss matrix
    losses = torch.zeros((xs.shape[0], n_slots, n_slots), device=device)
    for i in range(n_slots):
        for j in range(n_slots):
            losses[:, i, j] = (
                (xs[:, i].flatten(1) - xhs[:, j].flatten(1)).square().mean(1)
            )

    # Get indices from the Hungarian algorithm implemented in scipy
    indices_scipy = list(map(linear_sum_assignment, losses.cpu().detach().numpy()))
    indices_scipy = torch.LongTensor(np.array(indices_scipy)).to(device)
    smallest_cost_matrix = torch.stack(
        [
            losses[i][indices_scipy[i][0], indices_scipy[i][1]]
            for i in range(losses.shape[0])
        ]
    )

    return indices_scipy, smallest_cost_matrix


@pytest.mark.parametrize(
    "batch_size,n_slots",
    [
        (10, 4),
        (5, 3),
        (2, 5),
        (1, 6),
        # Add more combinations as needed...
    ],
)
def test_matching_indices(batch_size, n_slots):
    """
    Test that the indices returned by the Hungarian algorithm are correct and coincide with Jack's implementation.
    """
    # Define constants
    device = "cpu"

    # Generate distinct tensors for each slot
    xs = torch.stack(
        [
            torch.full((batch_size, 3, 64, 64), fill_value=i, device=device)
            for i in range(n_slots)
        ],
        dim=1,
    ).float()

    xhs = xs.clone()  # Clone to simulate predicted data
    for i in range(batch_size):
        perm = torch.randperm(n_slots)
        xhs[i] = xhs[i, perm, ...]
        # Save the permutation for each batch for later comparison
        if i == 0:
            perms = perm.unsqueeze(0)
        else:
            perms = torch.cat((perms, perm.unsqueeze(0)), dim=0)

    # Compute the indices and smallest cost matrix
    indices_scipy, smallest_cost_matrix = jacks_implementation(xs, xhs, n_slots, device)

    # Compare the recovered indices with original permutations
    assert torch.equal(
        torch.argsort(perms), indices_scipy[:, 1]
    )  # inverse permutation i.e. how you need to shuffle the predicted slots to match the true slots
    assert torch.isclose(
        smallest_cost_matrix, torch.zeros_like(smallest_cost_matrix)
    ).all()

    # flatten "images" to 1D vectors
    true_figures = xs.view(xs.shape[0], xs.shape[1], -1)
    predicted_figures = xhs.view(xhs.shape[0], xhs.shape[1], -1)

    matched_hungarian_slots_loss, indices_hungarian_slots_loss = hungarian_slots_loss(
        true_figures, predicted_figures, device
    )
    assert torch.isclose(
        matched_hungarian_slots_loss, torch.tensor(0.0, device=device), atol=1e-6
    )
    indices_hungarian_slots_loss = indices_hungarian_slots_loss.transpose(1, 2)

    # Assert that the indices are the same
    assert torch.equal(
        indices_scipy[:, 1], indices_hungarian_slots_loss[:, 1]
    )  # direct permutation i.e. how you need to shuffle the true slots to match the predicted slots
