"""
Implementation of evaluation metrics for point predictions (usually the mean of the predictive distribution).
"""

# Standard library imports
from typing import Tuple

# Third-party imports
import numpy as np
from src.evaluation.model_predictions import GroupedModelPrediction
import torch

# Local imports
from ..data import NPBDataset


def point_pred_eval(
    predictions: GroupedModelPrediction,
    test_data: NPBDataset,
) -> Tuple[float, float]:
    """
    Evaluate point predictions using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

    Aligns grouped predictions with the original target rows using
    ``predictions.original_indices``, then evaluates their mean prediction.

    Args:
        predictions: Grouped predictions containing row-wise means
        test_data: Test NPBDataset containing grouped target values

    Returns:
        Tuple containing:
            - MSE across all target rows
            - RMSE across all target rows
    """
    y_true = test_data.response.to_numpy().reshape(-1)
    y_pred = np.asarray(predictions.mean).reshape(-1)
    original_indices = np.asarray(predictions.original_indices).reshape(-1)

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Length mismatch: predictions.mean has {y_pred.shape[0]} rows, "
            f"but test_data.response has {y_true.shape[0]} rows."
        )
    if original_indices.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: predictions.original_indices has {original_indices.shape[0]} rows, "
            f"but predictions.mean has {y_pred.shape[0]} rows."
        )

    residuals = y_pred - y_true[original_indices]
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse


def squared_error(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared error between predictions and true values.

    Args:
        pred: Predicted values [n_target]
        truth: Tensor holding the true response values [n_target]

    Returns:
        Element-wise squared error between predictions and truth
    """
    return (pred - truth) ** 2
