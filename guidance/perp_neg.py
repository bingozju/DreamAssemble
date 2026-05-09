"""Perp-Neg perpendicular noise aggregator.

Reference: https://perp-neg.github.io/
"""

import torch


def get_perpendicular_component(x, y):
    """Component of ``x`` perpendicular to ``y``."""
    assert x.shape == y.shape
    return x - ((torch.mul(x, y).sum()) / max(torch.norm(y) ** 2, 1e-6)) * y


def batch_get_perpendicular_component(x, y):
    """Apply ``get_perpendicular_component`` along the batch dimension."""
    assert x.shape == y.shape
    return torch.stack(
        [get_perpendicular_component(x[i], y[i]) for i in range(x.shape[0])]
    )


def weighted_perpendicular_aggregator(delta_noise_preds, weights, batch_size):
    """Aggregate the main-positive prediction with weighted perpendicular negatives.

    Args:
        delta_noise_preds: ``[B * K, 4, H, W]`` differences between text-conditioned
            and unconditional predictions, with ``K`` prompts per direction.
        weights: ``[B * K]`` per-prompt scalar weights. ``weights[:B] == 1``.
        batch_size: ``B`` (the original batch size).
    """
    delta_noise_preds = delta_noise_preds.split(batch_size, dim=0)
    weights = weights.split(batch_size, dim=0)
    assert torch.all(weights[0] == 1.0)

    main_positive = delta_noise_preds[0]
    accumulated_output = torch.zeros_like(main_positive)
    for i, complementary in enumerate(delta_noise_preds[1:], start=1):
        idx = torch.abs(weights[i]) > 1e-4
        if idx.sum() == 0:
            continue
        accumulated_output[idx] += (
            weights[i][idx].reshape(-1, 1, 1, 1)
            * batch_get_perpendicular_component(complementary[idx], main_positive[idx])
        )

    assert accumulated_output.shape == main_positive.shape
    return accumulated_output + main_positive
