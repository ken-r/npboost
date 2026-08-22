from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelPrediction:
    """Prediction class for models that predict row-by-row.
    
    Args:
        mean: the mean predictions for each datapoint
        std: the std predictions for each datapoint
        group_ids: the group id for each datapoint
        fixed_effect: the fixed-effect prediction for each datapoint (only available for models with fixed-effect and random-effect components, e.g., LME)
        random_effect: the true random-effect prediction for each datapoint (only available for models with fixed-effect and random-effect components)
    """
    mean: np.ndarray
    std: Optional[np.ndarray]
    group_ids: np.ndarray
    fixed_effect: Optional[np.ndarray] = None
    random_effect: Optional[np.ndarray] = None


@dataclass
class GroupedModelPrediction:
    """Prediction class for models that predict group-by-group (NP, NPBoost)
    and that have predictive distributions available for each group.

    Args:
        mean: the mean predictions for each datapoint
        std: the std predictions for each datapoint
        group_ids: the group id for each datapoint
        original_indices: row positions of the datapoint in the original (ungrouped) input data
        predictive_distributions: a dictionary mapping group_id to the predictive distribution for that group
        fixed_effect: the fixed-effect prediction for each datapoint (available for models with fixed-effect and random-effect components, e.g., NPBoost)
        random_effect: the true random-effect prediction for each datapoint (available for models with fixed-effect and random-effect components)
    """
    mean: np.ndarray
    std: np.ndarray
    group_ids: np.ndarray
    original_indices: np.ndarray
    predictive_distributions: Dict[Any, Any]
    fixed_effect: Optional[np.ndarray] = None
    random_effect: Optional[np.ndarray] = None


    def reorder_to_original(self) -> "GroupedModelPrediction":
        """Reorder the predictions to match the order of the original data.
        This is useful for computing metrics, e.g. when the input response data is provided as a single array, but the model predicts group-by-group.
        """
        idx = np.argsort(self.original_indices)
        return GroupedModelPrediction(
            mean=self.mean[idx],
            std=self.std[idx],
            group_ids=self.group_ids[idx],
            original_indices=self.original_indices[idx],
            predictive_distributions=self.predictive_distributions,
            fixed_effect=self.fixed_effect[idx] if self.fixed_effect is not None else None,
            random_effect=self.random_effect[idx] if self.random_effect is not None else None,
        )
