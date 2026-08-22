"""
Implementation of the Continuous Ranked Probability Score (CRPS) for probabilistic evaluation.

Credits:
    This implementation is based on the pyro.ops.stats.crps_empirical function.
    Copyright (c) 2017-2019 Uber Technologies, Inc.
    SPDX-License-Identifier: Apache-2.0
    Original source: https://github.com/pyro-ppl/pyro
"""

# Standard library imports
from collections import defaultdict
import math
from typing import Dict, Any

# Third-party imports
import numpy as np
from src.evaluation.model_predictions import GroupedModelPrediction
import torch
from torch.distributions import Distribution
from torch.func import vmap
from scipy.stats import norm

# Local imports
from ..data.datasets import NPBDataset


def crps_eval(
    predictions: GroupedModelPrediction,
    test_data: NPBDataset,
    crps_samples: int,
    device: str = "cuda",
    return_samples: bool = False,
) -> float | tuple[float, Dict[Any, torch.Tensor]]:
    """
    Evaluate probabilistic predictions using the Continuous Ranked Probability Score (CRPS).

    Args:
        predictions: Grouped model predictions
        test_data: Test dataset containing grouped target values
        samples_crps: Number of samples to draw from each predictive distribution
        device: Device to perform computations on (defaults to cuda)
        return_samples: If True, return a dictionary of samples by group in addition to the mean CRPS score (useful for using the same samples for other metrics, like quantiles)

    Returns:
        Mean CRPS score across all groups, optionally with a dictionary of samples by group if return_samples is True.
    """
    device = torch.device(device)
    buckets = defaultdict(list)
    for group_id, pred_dist in predictions.predictive_distributions.items():
        buckets[pred_dist.mean.shape].append((group_id, pred_dist))

    total_crps = 0.0
    total_points = 0
    samples_by_group = {} if return_samples else None

    with torch.no_grad():
        for entries in buckets.values():
            # Extract predictive distribution parameters for groups with matching shape.
            means = torch.cat(
                [pred_dist.mean.to(device) for _, pred_dist in entries], dim=1
            )
            stds = torch.cat(
                [pred_dist.stddev.to(device) for _, pred_dist in entries], dim=1
            )

            # Draw samples from predictive distribution.
            y_sample = means.unsqueeze(0) + stds.unsqueeze(0) * torch.randn(
                (crps_samples,) + means.shape,
                device=device,
                dtype=means.dtype,
            )

            # Sample from the predictive distribution and reshape for CRPS computation.
            y_sample = (
                y_sample.flatten(0, 1)  # Flatten sample and latent sample dim
                .squeeze(-1)  # Remove trailing singleton response dimension
                .transpose(0, 1)  # Move batch_size to the leftmost dimension
            )  # [batch_size, n_z_samples * crps_samples, n_target]

            if return_samples:
                for batch_idx, (group_id, _) in enumerate(entries):
                    samples_by_group[group_id] = y_sample[batch_idx].detach().cpu()

            # Get true target values for this group and move to device.
            y_true = torch.stack(
                [
                    test_data.get_group(group_id)[1].squeeze(-1)  # (n_target)
                    for group_id, _ in entries
                ]
            ).to(device)

            batch_crps = vmap(crps_score)(y_sample, y_true)
            total_crps += batch_crps.sum().item()
            total_points += batch_crps.numel()

    mean_crps = total_crps / total_points
    if return_samples:
        return mean_crps, samples_by_group
    return mean_crps


def crps_score(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS) using the empirical formula.

    This implementation uses the empirical CRPS formula which is efficient for
    discrete samples from the predictive distribution.

    Args:
        pred: Samples from predictive distribution [num_samples, n_target]
        truth: Tensor holding the true response values [n_target]

    Returns:
        CRPS score for target [n_target]
    """

    # Get the number of samples from the first dimension
    num_samples = pred.shape[0]

    # Sort predictions along the sample dimension to prepare for empirical CRPS calculation
    pred_sorted = pred.sort(dim=0).values

    # Compute differences between consecutive sorted predictions
    diff = pred_sorted[1:] - pred_sorted[:-1]

    # Compute weights for the empirical CRPS formula
    # These weights account for the position of each sample in the sorted order
    left_counts = torch.arange(1, num_samples, device=pred.device, dtype=pred.dtype)
    right_counts = torch.arange(
        num_samples - 1, 0, -1, device=pred.device, dtype=pred.dtype
    )
    weight = left_counts * right_counts

    # Reshape weights to broadcast correctly with diff tensor
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    # Compute CRPS using the empirical formula:
    absolute_error_term = (pred_sorted - truth).abs().mean(0)
    diversity_term = (diff * weight).sum(0) / (num_samples**2)

    return absolute_error_term - diversity_term


def gaussian_crps(y_true, mu, sigma):
    """
    Calculates the average CRPS for a Gaussian distribution based on the
    provided formula.
    Does also work for degenerate cases where sigma contains zeros. 
    For those cases, the CRPS reduces to the absolute error.

    Args:
        y_true (np.ndarray): An array of observed true values.
        mu (np.ndarray): An array of predicted means of the Gaussian distributions.
        sigma (np.ndarray): An array of predicted standard deviations.

    Returns:
        The average CRPS value
    """

    # Ensure inputs are NumPy arrays for vectorized operations
    y_true = np.asarray(y_true)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    pos = sigma > 0
    crps_values = np.zeros_like(sigma)

    # Handle case when sigma is zero -> CRPS reduces to absolute error
    crps_values[~pos] = np.abs(y_true[~pos] - mu[~pos])
    
    # Standardize the observed values
    z = (y_true[pos] - mu[pos]) / sigma[pos]

    # Calculate the components of the formula
    term1 = 1 / np.sqrt(np.pi)
    term2 = 2 * norm.pdf(z)
    term3 = z * ((2 * norm.cdf(z)) - 1)

    # Calculate the CRPS for each individual prediction
    crps_values[pos] = (sigma[pos] * (term1 - term2 - term3)) * (-1)

    # Return the average of all individual CRPS values
    return np.mean(crps_values)


def quantile_crps(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    alphas: np.ndarray,
) -> float:
    """
    Approximate CRPS from predicted quantiles via integrated pinball loss.

    CRPS equals ``2 * integral_0^1 pinball_alpha(y - q_alpha) d_alpha`` for a predictive quantile function. 
    With finitely many quantiles, this uses a trapezoidal approximation and flat extrapolation to 0 and 1.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    quantiles = np.asarray(quantiles, dtype=float)
    alphas = np.asarray(alphas, dtype=float).reshape(-1)

    order = np.argsort(alphas)
    alphas = alphas[order]
    quantiles = quantiles[:, order]
    alphas = np.concatenate(([0.0], alphas, [1.0]))
    quantiles = np.concatenate([quantiles[:, :1], quantiles, quantiles[:, -1:]], axis=1)

    errors = y_true - quantiles
    pinball = np.maximum(
        alphas.reshape(1, -1) * errors,
        (alphas.reshape(1, -1) - 1.0) * errors,
    )
    crps_per_observation = 2.0 * np.trapezoid(pinball, alphas, axis=1)
    return float(np.mean(crps_per_observation))
