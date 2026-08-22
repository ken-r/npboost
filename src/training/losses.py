"""
Loss functions for NP based models.
"""

import math

# Third-party imports
import torch
import torch.nn as nn
from torch import logsumexp
from torch.distributions import Independent


class NPMLLoss(nn.Module):
    """
    Neural Process Marginal Likelihood (NPML) loss function.

    This loss function was originally introduced by Foong et al. (2020) as an alternative
    to the standard variational bound introduced in Garnelo et al. (2018).
    """

    def __init__(self):
        """Initialize NPML loss function with no additional parameters. Inherit from nn.Module"""
        super().__init__()

    def forward(self, predictions: Independent, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Neural Process Marginal Likelihood (NPML) loss.

        Args:
            prediction: Distribution with batch_shape [n_z_samples, batch_size, n_target]
            targets: Tensor of ground truth target values with size [batch_size, n_target]

        Returns:
            torch.Tensor: Scalar NPML loss value for each batch [batch_size]

        Note:
            The loss value for each batch is an approximation for the log likelihood of the
            target points under the model that produced the predictive distribution. The loss
            value is not invariant to different number of target points.
        """

        # Compute log-likelihood for each latent sample and target point
        log_likelihood = predictions.log_prob(targets)

        # Extract number of latent samples (z) for normalization.
        n_z_samples = float(log_likelihood.shape[0])
        norm = math.log(n_z_samples)

        # Sum log-likelihood over target points for each batch and latent sample
        summed_log_likelihood = log_likelihood.sum(dim=-1)

        # Compute log-marginal likelihood using logsumexp for numerical stability
        log_marginal_likelihood = logsumexp(summed_log_likelihood, dim=0) - norm

        # Return negative log-marginal likelihood for each function in the batch
        return -log_marginal_likelihood
