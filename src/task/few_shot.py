"""Few-shot prediction scenario splitter.

For few-shot prediction, we hold out complete groups and mark support/target rows within them.
All observations of the training groups are used for model training.
"""

import numpy as np

from src.constants import (
    CONTEXT_ROLE_COLUMN_NAME,
    FEW_SHOT_TASK_NAME,
    GROUPING_COLUMN_NAME,
    SPLIT_COLUMN_NAME,
    SUPPORT_ROLE_NAME,
    TARGET_ROLE_NAME,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
)
from src.task.base_task import BaseTask
from src.utils.helpers import set_seed


class FewShot(BaseTask):
    """Hold out complete groups and mark support/target rows within them.

    Args:
        n_validation_groups: Number of groups assigned to validation.
        n_test_groups: Number of groups assigned to test.
        n_support_validation: Number of support rows sampled in each validation
            group.
        n_support_test: Number of support rows sampled in each test group.
    """

    def __init__(
        self,
        n_validation_groups,
        n_test_groups,
        n_support_validation,
        n_support_test,
    ):
        self.n_validation_groups = n_validation_groups
        self.n_test_groups = n_test_groups
        self.n_support_validation = n_support_validation
        self.n_support_test = n_support_test

    def split_data(self, df, seed):
        """Assign groups to splits and label held-out support/target rows.

        Args:
            df: Input data frame containing ``GROUPING_COLUMN_NAME``.
            seed: Random seed used for group and support-row sampling.

        Returns:
            A copied data frame with columns containing split and support/target assignments in the columns ``SPLIT_COLUMN_NAME`` and ``CONTEXT_ROLE_COLUMN_NAME`` for the validation and test groups.
        """
        # Create a copy of the data, since we will modify it
        df = df.copy()
        set_seed(seed)

        # Initialize the split column
        df[SPLIT_COLUMN_NAME] = TRAIN_SPLIT_NAME
        df[CONTEXT_ROLE_COLUMN_NAME] = None

        unique_groups = np.unique(df[GROUPING_COLUMN_NAME])
        group_perm_indices = np.random.permutation(unique_groups)
        validation_groups = group_perm_indices[:self.n_validation_groups]
        test_groups = group_perm_indices[
            self.n_validation_groups : self.n_validation_groups + self.n_test_groups
        ]

        for group in validation_groups:
            group_indices = df[df[GROUPING_COLUMN_NAME] == group].index
            
            df.loc[group_indices, SPLIT_COLUMN_NAME] = VALIDATION_SPLIT_NAME
            df.loc[group_indices, CONTEXT_ROLE_COLUMN_NAME] = TARGET_ROLE_NAME
            
            support_indices = np.random.choice(
                group_indices, size=self.n_support_validation, replace=False
            )
            
            df.loc[support_indices, CONTEXT_ROLE_COLUMN_NAME] = SUPPORT_ROLE_NAME

        # Process test groups with the same support/target split structure.
        for group in test_groups:
            group_indices = df[df[GROUPING_COLUMN_NAME] == group].index
            
            df.loc[group_indices, SPLIT_COLUMN_NAME] = TEST_SPLIT_NAME
            df.loc[group_indices, CONTEXT_ROLE_COLUMN_NAME] = TARGET_ROLE_NAME
            
            support_indices = np.random.choice(
                group_indices, size=self.n_support_test, replace=False
            )
            
            df.loc[support_indices, CONTEXT_ROLE_COLUMN_NAME] = SUPPORT_ROLE_NAME

        return df

    @property
    def name(self):
        return FEW_SHOT_TASK_NAME

    @property
    def training_context_size(self) -> float:
        """Return the test-time support-set size so training can be done in the way that the model will be evaluated."""
        return self.n_support_test
