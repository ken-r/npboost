"""In-context prediction scenario splitter.

For in-context prediction, each group is used for training.
However, within each group, a fraction of the rows are assigned to validation and test splits.
"""

import logging

import numpy as np

from src.constants import (
    GROUPING_COLUMN_NAME,
    IN_CONTEXT_TASK_NAME,
    SPLIT_COLUMN_NAME,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
)
from src.task.base_task import BaseTask
from src.utils.helpers import set_seed

logger = logging.getLogger(__name__)


class InContext(BaseTask):
    """Create within-group train/validation/test splits for in-context prediction.

    Each group contributes observations to all available splits: validation and
    test rows are sampled within group, and the remaining rows stay in the
    training split. 
    The actual context/target split size used during training is set by ``training_context_size``.

    Args:
        training_context_size: Fraction of each training function/group to be used as
            context during neural process training.
        validation_size: Fraction of each group assigned to validation.
        test_size: Fraction of each group assigned to test.
    """

    def __init__(
        self,
        training_context_size: float = 0.5,
        validation_size: float = 0.1,
        test_size: float = 0.1,
    ):
        self._training_context_size = training_context_size
        self.validation_size = validation_size
        self.test_size = test_size

    def split_data(self, df, seed):
        """Assign each row to train, validation, or test within its group.

        Args:
            df: Input data frame containing ``GROUPING_COLUMN_NAME``.
            seed: Random seed used to sample validation and test rows.

        Returns:
            A copied data frame with ``SPLIT_COLUMN_NAME`` assigned to train, validation, or test.
        """
        # Create a copy of the data, since we will modify it
        df = df.copy()
        set_seed(seed)
        unique_groups = df[GROUPING_COLUMN_NAME].unique()
        df[SPLIT_COLUMN_NAME] = TRAIN_SPLIT_NAME
        validation_indices = []
        test_indices = []
        for group in unique_groups:
            group_indices = df[df[GROUPING_COLUMN_NAME] == group].index
            n_samples = len(group_indices)
            
            n_validation = int(n_samples * self.validation_size)
            n_test = int(n_samples * self.test_size)

            if n_validation == 0:
                logger.warning(
                    "Group %s has too few samples (%s) for the specified "
                    "validation size (%s). Skipping validation split for this group.",
                    group,
                    n_samples,
                    self.validation_size,
                )
            if n_test == 0:
                logger.warning(
                    "Group %s has too few samples (%s) for the specified "
                    "test size (%s). Skipping test split for this group.",
                    group,
                    n_samples,
                    self.test_size,
                )

            perm_indices = np.random.permutation(group_indices)
            
            validation_indices.extend(perm_indices[:n_validation])
            test_indices.extend(perm_indices[n_validation : n_validation + n_test])
            
        df.loc[validation_indices, SPLIT_COLUMN_NAME] = VALIDATION_SPLIT_NAME
        df.loc[test_indices, SPLIT_COLUMN_NAME] = TEST_SPLIT_NAME

        return df

    @property
    def name(self):
        return IN_CONTEXT_TASK_NAME

    @property
    def training_context_size(self) -> float:
        """Return the training context fraction. Training is done with this fraction used for the context set."""
        return self._training_context_size
