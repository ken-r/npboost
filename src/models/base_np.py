"""
Abstract base class for Neural Process model implementations.
"""

# Standard library imports
from typing import Tuple
from abc import ABC, abstractmethod

# Third-party imports
import torch.nn as nn
from torch import Tensor
from torch.distributions import Independent


class BaseNP(nn.Module, ABC):
    """
    Abstract base class for Neural Process model implementations.

    Defines the standard interface that all Neural Process variants must implement.
    """

    def __init__(self) -> None:
        """Initialize the base Neural Process model."""
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x_context: Tensor,
        y_context: Tensor,
        x_target: Tensor = None,
    ) -> Tuple[Independent, Independent]:
        """
        Forward pass through the Neural Process model.

        Implements the core Neural Process computation: encode context data into
        a representation, then decode predictions at target locations. This method
        must be implemented by all subclasses.

        Args:
            x_context: Context input locations/features
            y_context: Context target values/observations
            x_target: Target input locations where predictions are needed

        Returns:
            Tuple containing:
                - predictive: Predictive distribution at target locations
                - latent: Latent distribution representing the learned function
        """
        # This method must be implemented by subclasses
        pass
