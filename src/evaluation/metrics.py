import numpy as np
import pandas as pd
from src.constants import GROUPING_COLUMN_NAME
import torch

from src.data.datasets import BaseDataset, NPBDataset
from src.evaluation.model_predictions import GroupedModelPrediction, ModelPrediction
from src.evaluation.crps import crps_score, gaussian_crps, quantile_crps

def groupwise_metrics(
    predictions: ModelPrediction | GroupedModelPrediction,
    target_data: BaseDataset | NPBDataset,
    group_ids: np.ndarray | None = None,
    eval_samples: int = 20,
) -> pd.DataFrame:
    """Compute group-wise metrics (RMSE and CRPS) for model predictions.
    This is useful for plotting, when you want to visualize the performance of a model on each group separately.
    """
    if isinstance(predictions, GroupedModelPrediction):
        # Reorder predictions to match the data order in target_data.response
        pred = predictions.reorder_to_original()
    else:
        pred = predictions

    y_true = np.asarray(target_data.response).reshape(-1)
    mean = np.asarray(pred.mean).reshape(-1)

    # If no group id provided, we calculate metrics for all groups in the prediction data.
    if group_ids is None:
        group_ids = np.asarray(pred.group_ids)
    else:
        group_ids = np.asarray(group_ids)

    if len(y_true) != len(mean):
        raise ValueError(
            f"Length mismatch: target_data.response has {len(y_true)} rows, "
            f"but predictions.mean has {len(mean)} rows."
        )

    if len(group_ids) != len(y_true):
        raise ValueError(
            "group_ids must have one entry per row in target_data.response; "
            "pass per-row group labels, not a list of selected groups."
        )

    std = None
    if hasattr(pred, "std") and pred.std is not None:
        std = np.asarray(pred.std).reshape(-1)

    group_metrics = []

    for group in pd.unique(group_ids):
        mask = group_ids == group
        y_g = y_true[mask]
        mean_g = mean[mask]

        rmse_g = float(np.sqrt(np.mean((mean_g - y_g) ** 2)))

        if hasattr(pred, "predictive_distributions") and pred.predictive_distributions is not None:
            # Evaluate the CRPS using samples from the predictive distribution for this group
            dist = pred.predictive_distributions[group]
            sample = dist.sample(torch.Size([eval_samples]))
            sample = (
                torch.as_tensor(sample)
                .detach()
                .to("cpu")
                .reshape(-1, int(mask.sum()))
            )
            y_t = torch.as_tensor(y_g, device="cpu")
            crps_g = float(crps_score(sample, y_t).mean().item())
        elif (
            # Evaluate the CRPS using quantiles
            hasattr(pred, "quantiles")
            and pred.quantiles is not None
            and hasattr(pred, "quantile_alphas")
            and pred.quantile_alphas is not None
        ):
            crps_g = quantile_crps(
                y_g,
                np.asarray(pred.quantiles)[mask],
                np.asarray(pred.quantile_alphas),
            )
        else:
            # Evaluate the CRPS using the Gaussian approximation
            std_g = std[mask]
            crps_g = float(np.mean(gaussian_crps(y_g, mean_g, std_g)))

        group_metrics.append(
            {
                GROUPING_COLUMN_NAME: group,
                "rmse": rmse_g,
                "crps": crps_g,
                "n": int(mask.sum()),
            }
        )

    return pd.DataFrame(group_metrics)
