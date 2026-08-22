"""Base abstraction for splitting according to prediction task (prediction scenario)."""

from abc import ABC, abstractmethod
from typing import Union


class BaseTask(ABC):
    """
    A ``BaseTask`` describes the prediction scenarios for which we want to apply the model.
    Here, ``task`` means prediction scenario. 
    It should not be confused with the paper's terminology, where a task corresponds to a ``group`` in grouped data / mixed-effects notation.
    """

    @abstractmethod
    def split_data(self, df, seed):
        """Split raw data into train, validation, and test subsets.

        Args:
            df: Input data frame. Implementations expect a grouping column and
                may require additional feature or metadata columns.
            seed: Random seed used for data splits.

        Returns:
            A copied data frame with task-specific splits.
        """
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def training_context_size(self) -> Union[int, float]:
        """Context size that will be used for model training.

        In-context tasks return a fraction (context proportion to be used during training). 
        Few-shot tasks return an integer (the exact number of context points to be used during training).
        """
        pass
