"""
General utility functions for the NPBoost project.

This module provides helper functions for common tasks such as setting random seeds for
reproducibility, splitting data, and managing the loading/saving of datasets.
"""

# Standard library imports
from contextlib import contextmanager
import pickle
import random
from pathlib import Path
from typing import Dict, Tuple, Union
import logging.config

# Third-party imports
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor



def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Sets the seed for Python's `random`, `NumPy`, and `PyTorch` (both CPU and
    GPU). It also configures PyTorch's CUDA backend for deterministic
    operations, which may have a minor performance cost.

    Args:
        seed: The integer value to use for all random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure PyTorch for deterministic behavior
    # Set to False for better performance. For the final run, we should set it to True.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # Warn if non-deterministic functions are used


@contextmanager
def isolated_torch_rng(seed: int, device: Union[str, torch.device, None] = None):
    """Use a temporary, seeded random number generator stream while preserving the original global random state.
    
    This context manager allows you to perform random operations in PyTorch with a specific seed,
    without affecting the global random state. This is useful for reproducibility if for example, you want to generate intermediate
    plots (that need samples), but you do not want to experiment's performance to change when you run it without intermediate plots."""
    # Save the current random state on entry and restore it on exit.
    # Any random operations inside this block will not affect the main program.
    devices = []
    if device is not None:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            # Extract the raw integer index (e.g., 0), which fork_rng requires instead of a device object.
            devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]
    elif torch.cuda.is_available() and torch.cuda.is_initialized():
        # If the user didn't specify a device but CUDA is active, protect all available GPUs.
        devices = list(range(torch.cuda.device_count()))

    with torch.random.fork_rng(devices=devices):
        # Apply the temporary seed only for the duration of this context block.
        torch.manual_seed(seed)
        yield


def split_context(
    x: Tensor,
    y: Tensor,
    context_size: Union[int, float],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Randomly split data into context and target sets for an NP.

    Args:
        x: Input features tensor of shape (batch, num_points, dims).
        y: Target values tensor of shape (batch, num_points, dims).
        context_size: The fraction of points to use as context (0.0, 1.0).

    Returns:
        A tuple containing:
        - x_context (Tensor): Context input features.
        - y_context (Tensor): Context target values.
        - x_target (Tensor): Target input features.
        - y_target (Tensor): Target values for prediction.
    """
    # Calculate the number of context points
    n_points = x.shape[1]

    if context_size < 1.0:
        n_context = max(1, int(n_points * context_size))
    else:
        if context_size >= n_points:
            raise ValueError(f"n_context={context_size} >= n_points={n_points}")
        n_context = context_size

    # Randomly select indices for the context set without replacement
    perm_indices = torch.randperm(n_points, device=x.device)
    context_indices = perm_indices[:n_context]
    target_indices = perm_indices[n_context:]


    # Split the data using the indices and the mask
    x_context = x[:, context_indices, :]
    y_context = y[:, context_indices, :]
    x_target = x[:, target_indices, :]
    y_target = y[:, target_indices, :]

    return x_context, y_context, x_target, y_target


def setup_logging(default_level=logging.INFO):
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': default_level,
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': 'WARNING',
            },
            'alembic': {
                'level': 'WARNING',
                'propagate': False,
            },
            'src': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
            '__main__': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False,
            },
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)
