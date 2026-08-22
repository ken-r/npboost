"""Dataset class and data loaders for our experiments.

This module contains classes for basic datasets and also for np-style datasets that allow batched sampling of entire groups.
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from src.constants import (
    CONTEXT_ROLE_COLUMN_NAME,
    ENCODED_COLUMN_PREFIX,
    FEW_SHOT_TASK_NAME,
    FIXED_EFFECT_PART_NAME,
    FIXED_ONLY_COLUMN_PREFIX,
    GROUPING_COLUMN_NAME,
    IN_CONTEXT_TASK_NAME,
    INTERCEPT_COLUMN_NAME,
    NOISE_PART_NAME,
    RANDOM_EFFECT_PART_NAME,
    RESPONSE_COLUMN_NAME,
    SPLIT_COLUMN_NAME,
    SUPPORT_ROLE_NAME,
    TARGET_ROLE_NAME,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
)
from src.task.base_task import BaseTask


logger = logging.getLogger(__name__)

NO_FEATURE_COLUMNS = [FIXED_EFFECT_PART_NAME, RANDOM_EFFECT_PART_NAME, NOISE_PART_NAME]


class BaseDataset:
    """Tabular split data shared by neural and non-neural model adapters."""

    def __init__(
        self,
        features: pd.DataFrame,
        response: pd.Series,
        grouping_var: pd.Series,
        fixed_effect_part: Optional[pd.Series] = None,
        random_effect_part: Optional[pd.Series] = None,
        noise_part: Optional[pd.Series] = None,
    ) -> None:
        self.features = features
        self.response = response
        self.grouping_var = grouping_var
        self.fixed_effect_part = fixed_effect_part
        self.random_effect_part = random_effect_part
        self.noise_part = noise_part


class ModelDataAdapter:
    """Convert ``BaseDataset`` objects to the inputs expected by each model.

    Models that have a fixed-effect and random effect component receive all features for the fixed effect,
    but non-encoded features and additionally marked features are dropped for the random effect.
    All returned arrays preserve the row order of the input ``BaseDataset``.
    """

    @staticmethod
    def to_lme_data(ds: BaseDataset):
        """Transform a ``BaseDataset`` into arrays for fitting an LME model.

        Returns:
            X_fixed: Feature values to use for fixed effect.
            X_random: Features values to use for random effect.
            group_data: Group ids.
            drop_intercept: Flag for random-effect intercept handling.
            y: Responses.
        """
        X = ds.features.copy()
        fixed_cols = list(X.columns)
        X[INTERCEPT_COLUMN_NAME] = 1.0
        fixed_effect_columns = fixed_cols + [INTERCEPT_COLUMN_NAME]
        random_effect_columns = [
            col for col in fixed_cols
            if not str(col).startswith(ENCODED_COLUMN_PREFIX)
            and not str(col).startswith(FIXED_ONLY_COLUMN_PREFIX)
        ] + [INTERCEPT_COLUMN_NAME]
        logger.info(f"""Extracting data for LME.\n
                    Using fixed effects columns: {fixed_effect_columns}\n
                    Using random effects columns: {random_effect_columns}""")
        return (
            X[fixed_effect_columns].values,
            X[random_effect_columns].values,
            ds.grouping_var.values,
            [True],
            ds.response.values,
        )


    @staticmethod
    def to_gplinear_data(ds: BaseDataset):
        """Transform a ``BaseDataset`` into arrays for fitting a GPLinear model.
        
        Returns:
            X_fixed: Feature values to use for fixed effect (including intercept).
            gp_coords: Feature values to use for random effect.
            group_data: Group ids.
            y: Responses.
        """
        X = ds.features.copy()
        fixed_cols = list(X.columns)
        X[INTERCEPT_COLUMN_NAME] = 1.0
        fixed_effect_columns = fixed_cols + [INTERCEPT_COLUMN_NAME]
        random_effect_columns = [
            col for col in fixed_cols
            if not str(col).startswith(ENCODED_COLUMN_PREFIX)
            and not str(col).startswith(FIXED_ONLY_COLUMN_PREFIX)
        ]
        logger.info(f"""Extracting data for GPLinear.\n
                    Using fixed effects columns: {fixed_effect_columns}\n
                    Using random effects columns: {random_effect_columns}\n""")
        return X[fixed_effect_columns].values, X[random_effect_columns].values, ds.grouping_var.values, ds.response.values

    @staticmethod
    def to_np_data(ds: BaseDataset):
        """Return an ``NPBDataset`` for regular Neural Process training.

        The returned ``NPBDataset`` uses all feature columns for NP fitting.
        """
        logger.info(f"""Converting to NPBDataset.\n
                    Using feature columns: {ds.features.columns}""")
        return NPBDataset(ds.features.copy(), ds.grouping_var, ds.response)

    @staticmethod
    def to_npboost_data(ds: BaseDataset):
        """Return an ``NPBDataset`` for regular Neural Process training.

        The returned ``NPBDataset`` uses only non-one-hot encoded features and non-fixed-only features
        for NP fitting.
        """
        np_feature_columns = [
            col for col in ds.features.columns
            if not str(col).startswith(ENCODED_COLUMN_PREFIX)
            and not str(col).startswith(FIXED_ONLY_COLUMN_PREFIX)
        ]
        np_features = ds.features[np_feature_columns]
        logger.info(f"""Converting to NPBDataset for NPBoost.\n
                    Using fixed effect columns: {ds.features.columns}\n
                    Using NP columns: {np_features.columns}""")
        return NPBDataset(
            features=ds.features.copy(),
            np_features=np_features.copy(),
            grouping_var=ds.grouping_var,
            response=ds.response,
        )


class FunctionSampler:
    """
    Custom batch sampler for Neural Process training that groups function samples by size.

    This sampler enables efficient training with function data where functions have different
    lengths by ensuring that all functions in a batch have the same size. This allows for
    proper tensor batching without padding and improves training efficiency.
    """

    def __init__(
        self, dataset: "NPBDataset", batch_size: int, shuffle: bool = True
    ) -> None:
        """
        Initialize the FunctionSampler.

        Args:
            dataset: NPBDataset instance to sample from
            batch_size: Number of function samples per batch
            shuffle: Whether to randomize the order of batches and samples within batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group dataset indices by function size for efficient batching
        # This creates a mapping: {function_size: [dataset_indices]}
        self.size_groups: Dict[int, List[int]] = defaultdict(list)
        for idx, group_id in enumerate(dataset.unique_groups):
            size = len(dataset._group_indices[group_id])
            self.size_groups[size].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of dataset indices grouped by function size.

        Yields:
            List of dataset indices for each batch
        """
        all_batches: List[List[int]] = []

        # Process each size group separately to ensure same-size batching
        for indices in self.size_groups.values():
            indices_copy = indices.copy()
            if self.shuffle:
                random.shuffle(indices_copy)

            # Create batches for this size group using sliding window
            all_batches.extend(
                indices_copy[i : i + self.batch_size]
                for i in range(0, len(indices_copy), self.batch_size)
            )

        # Shuffle the order of batches across different sizes if requested
        if self.shuffle:
            random.shuffle(all_batches)

        # Yield each batch as a list of dataset indices
        yield from all_batches

    def __len__(self) -> int:
        """
        Calculate the total number of batches across all size groups.

        Returns:
            Total number of batches that will be generated by this sampler
        """
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.size_groups.values()
        )

# TODO: Replace 'target' with 'response' in the following class. Target always refers to the response variable in this class,
# but it can be confusing since the term 'target' is also used for context-target sets for NPs...
class NPBDataset(Dataset):
    """
    Neural Process Boosting (NPB) dataset class for function-based learning.

    This PyTorch Dataset implementation is designed specifically for Neural Process training
    within the NPBoost framework. It organizes tabular data into task-specific function
    realizations, where each task corresponds to one value of the grouping variable.

    Terminology:
        In the paper, groups are called tasks.

    Key Features:
        - Groups observations by a grouping variable into function realizations
        - Enables updating the response (e.g. for residualizing the response as in NPBoost algorithm)
        - Provides PyTorch-compatible tensor outputs

    Note:
        - __len__ returns the number of function realizations (groups)
        - __getitem__ returns feature-response tensors for a specific group
    """

    def __init__(
        self,
        features: pd.DataFrame,
        grouping_var: pd.Series,
        response: Optional[pd.Series] = None,
        np_features: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize NPBDataset with features, responses, and grouping information.

        Args:
            features: Input features as pandas DataFrame with shape (n_observations, n_features)
            grouping_var: Group membership for each observation as pandas Series
            response: Response values as pandas Series. If None, creates zeros for prediction
            np_features: Optional feature view used by the Neural Process. If None, it uses all features

        Raises:
            ValueError: If features, np_features, and grouping_var have incompatible lengths
        """
        if len(features) != len(grouping_var):
            raise ValueError("features and grouping_var must have the same length")
        if np_features is not None and len(np_features) != len(grouping_var):
            raise ValueError("np_features and grouping_var must have the same length")
        
        self.features = features.reset_index(drop=True)
        self.np_features = (np_features if np_features is not None else features).reset_index(drop=True)
        self.grouping_var = grouping_var.reset_index(drop=True)
        self.response = (response if response is not None else pd.Series(0, index=features.index)).reset_index(drop=True)
        self.categorical_feature_column_names = [col for col in features.columns if col.startswith(ENCODED_COLUMN_PREFIX)]

        # Create grouped data structure for NP training and prediction
        self._group_indices = self.grouping_var.groupby(self.grouping_var).indices
        self.unique_groups = sorted(list(self._group_indices.keys()))
        self._create_grouped_data()

    def _create_grouped_data(self) -> None:
        """
        Create grouped data structure optimized for Neural Process training and prediction.

        This method organizes the dataset into groups based on the grouping variable,
        converting pandas DataFrames and Series into PyTorch tensors for efficient
        Neural Process operations. Each group contains features, responses, and original
        row indices as separate tensors.
        The original row indices are preserved to allow for mapping back to the original dataset order as it appeared in the original dataset.
        """

        # Initialize the dictionary that holds the grouped data
        self.groups = {}
        features_np = self.np_features

        # Loop through groups and create grouped data
        for group_id, indices in self._group_indices.items():

            # Get the grouped data for this specific group
            group_features = features_np.iloc[indices].values
            group_responses = self.response.iloc[indices].values

            # Store as tensors for Neural Process compatibility
            self.groups[group_id] = {
                "features": torch.tensor(group_features, dtype=torch.float32),
                "responses": torch.tensor(group_responses, dtype=torch.float32),
                "original_indices": torch.tensor(indices, dtype=torch.long)
            }

    def get_group(self, group_id: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve feature and response tensors for a specific group.

        Args:
            group_id: Identifier of the group to retrieve (can be any hashable type)

        Returns:
            Tuple containing:
                - features: Feature tensor of shape (n_points, n_features)
                - responses: Response tensor of shape (n_points, 1)
                - original_indices: Tensor of original row indices for this group

        Raises:
            KeyError: If group_id is not found in the dataset
        """
        if group_id not in self._group_indices:
            raise KeyError(f"Group {group_id} not found.")
            
        group_data = self.groups[group_id]
        return (
            group_data["features"], 
            group_data["responses"].unsqueeze(-1),  # Add dimension for consistency
            group_data["original_indices"],
        )

    def update_responses(self, new_responses: np.ndarray) -> None:
        """Replace responses by new values."""
        self.response = pd.Series(new_responses, index=self.response.index)
        self._create_grouped_data()  # Recreate grouped data structure with updated responses.

    def __len__(self) -> int:
        """
        Return the number of function realizations (groups) in the dataset.

        Note: This returns the number of groups, not the total number of observations,
        which is consistent with Neural Process training where each group represents
        one function realization.

        Returns:
            Number of unique groups in the dataset
        """
        return len(self.unique_groups)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a function realization (group) from the dataset by index.

        This method is designed for Neural Process training where each group represents
        a different function realization. It returns the features and responses for all
        observations within the specified group.

        Args:
            index: Index of the group to retrieve (0 <= index < len(self))

        Returns:
            Tuple containing:
                - features: Feature tensor of shape (n_points, n_features)
                - responses: Response tensor of shape (n_points, 1)
                - original_indices: Tensor of original row indices for this group

        Raises:
            IndexError: If index is out of range
        """
        group_id = self.unique_groups[index]
        return self.get_group(group_id)

    def __repr__(self) -> str:
        return f"NPBDataset(n_functions={len(self)}, n_observations={len(self.features)})"


class PairedFunctionSampler:
    """
    Batch sampler for PairedNPBDataset.

    Bins dataset indices by (context_size, target_size) so that all pairs in a
    batch have identical shapes.
    Pairs whose size signature is unique form a batch of size 1.
    """

    def __init__(
        self,
        dataset: "PairedNPBDataset",
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Bin indices by (context_size, target_size)
        size_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            key = (dataset.context_size(idx), dataset.target_size(idx))
            size_groups[key].append(idx)

        self._batches: List[List[int]] = []
        for indices in size_groups.values():
            for i in range(0, len(indices), batch_size):
                self._batches.append(indices[i : i + batch_size])

    def __iter__(self):
        batches = self._batches
        if self.shuffle:
            batches = self._batches.copy()
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)

class PairedNPBDataset(Dataset):
    """
    Dataset of (context, target) group pairs for batched NP validation.

    Each item is one group_id's context observations paired with that group_id's
    target observations.
    """

    def __init__(
        self,
        context_npbd: "NPBDataset",
        target_npbd: "NPBDataset",
    ) -> None:
        self.context_npbd = context_npbd
        self.target_npbd = target_npbd
        
        if self.context_npbd.unique_groups != self.target_npbd.unique_groups:
            raise ValueError("Context and target datasets must have the same group ids.")

        self.unique_groups = target_npbd.unique_groups

    def __len__(self) -> int:
        return len(self.unique_groups)

    def __getitem__(self, index: int):
        group_id = self.unique_groups[index]
        x_context, y_context, indices_context = self.context_npbd.get_group(group_id)   # [context_size, D], [context_size, 1]
        x_target, y_target, indices_target = self.target_npbd.get_group(group_id)    # [target_size, D], [target_size, 1]
        return x_context, y_context, indices_context, x_target, y_target, indices_target

    def context_size(self, index: int) -> int:
        group_id = self.unique_groups[index]
        return len(self.context_npbd._group_indices[group_id])

    def target_size(self, index: int) -> int:
        group_id = self.unique_groups[index]
        return len(self.target_npbd._group_indices[group_id])
    

def paired_collate_fn(batch):
    """
    Collate (x_context, y_context, x_target, y_target) tuples into batched tensors.

    PairedFunctionSampler guarantees all items in a batch share the same
    context size and the same target size.
    However, context size != target size is possible.
    """
    x_context, y_context, indices_context, x_target, y_target, indices_target = zip(*batch)
    return (
        torch.stack(x_context),  # [B, context_size, D]
        torch.stack(y_context),  # [B, context_size, 1]
        torch.stack(indices_context),  # [B, context_size]
        torch.stack(x_target),  # [B, target_size, D]
        torch.stack(y_target),  # [B, target_size, 1]
        torch.stack(indices_target),  # [B, target_size]
    )


class GradientSplitDataset(Dataset):
    """
    Dataset of context/target splits used to compute NPBoost gradients.

    Each item is one group's gradient split. The existing PairedFunctionSampler can
    batch this dataset because it exposes context_size() and target_size().
    The generated splits mirror the active prediction scenario: in-context uses
    held-out folds within a group, while few-shot uses k support points and predicts
    the remaining points.
    """

    def __init__(self, npbd: "NPBDataset", task: BaseTask) -> None:
        self.npbd = npbd
        self.task = task
        self.items = []

        for group_id in npbd.unique_groups:
            x, y, original_indices = npbd.get_group(group_id)
            splits, normalizer = self._get_splits(len(x))
            for train_idx, target_idx in splits:
                if len(target_idx) == 0:
                    logger.warning("Skipping empty target split during gradient calculation.")
                    continue
                self.items.append(
                    {
                        "group_id": group_id,
                        "x": x,
                        "y": y,
                        "train_idx": torch.tensor(train_idx, dtype=torch.long),
                        "target_idx": torch.tensor(target_idx, dtype=torch.long),
                        "original_indices": original_indices,
                        "normalizer": normalizer,
                    }
                )
    def _get_splits(self, n_total_group: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
        """Create context/target index splits for one group's NPBoost gradient.

        Returns:
            A list of ``(context_idx, target_idx)`` splits and the gradient
            normalizer. In-context tasks use K-fold held-out targets so each row
            is targeted once. Few-shot tasks rotate folds of ``k`` support rows;
            because each observation appears in multiple target sets, the
            normalizer is the number of times that an observation is used as a target.
        """
        splits = []

        if self.task.name == IN_CONTEXT_TASK_NAME:
            # Rotate held-out folds so every observation contributes as a target once.
            target_ratio = 1.0 - self.task.training_context_size
            n_splits = int(round(1.0 / target_ratio))
            kf = KFold(n_splits=n_splits, shuffle=True)
            for train_idx, target_idx in kf.split(range(n_total_group)):
                splits.append((train_idx, target_idx))
            return splits, 1.0

        if self.task.name == FEW_SHOT_TASK_NAME:
            # Condition on k support points and predict all remaining points.
            k_context = self.task.training_context_size
            indices = np.arange(n_total_group)
            np.random.shuffle(indices)

            for i in range(0, n_total_group, k_context):
                train_idx = indices[i : i + k_context]
                target_idx = np.setdiff1d(indices, train_idx)
                splits.append((train_idx, target_idx))

            return splits, float(len(splits) - 1)

        raise ValueError(f"Unknown task name: {self.task.name}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        train_idx = item["train_idx"]
        target_idx = item["target_idx"]
        return (
            item["group_id"],
            item["x"][train_idx],
            item["y"][train_idx],
            item["x"][target_idx],
            item["y"][target_idx],
            item["original_indices"][target_idx],
            item["normalizer"],
        )

    def context_size(self, index: int) -> int:
        return len(self.items[index]["train_idx"])

    def target_size(self, index: int) -> int:
        return len(self.items[index]["target_idx"])


def gradient_split_collate_fn(batch):
    """Collate gradient splits into the tensor shapes expected by NPBoost."""
    group_id, x_context, y_context, x_target, y_target, original_indices, normalizer = zip(*batch)
    return (
        group_id,
        torch.stack(x_context),
        torch.stack(y_context),
        torch.stack(x_target),
        torch.stack(y_target),
        torch.stack(original_indices),
        torch.tensor(normalizer, dtype=torch.float32),
    )

class DataBundle:
    """Generic bundle for holding train/validation/test data for one experiment.
    For in-context task, validation and test data are available, but *support and *query are None.
    For few-shot task, *support and *query are available, but validation and test data are None. 
    """
    def __init__(self, train: BaseDataset, validation: BaseDataset=None, test: BaseDataset=None,
                 validation_support: BaseDataset=None, validation_query: BaseDataset=None,
                 test_support: BaseDataset=None, test_query: BaseDataset=None):
        self.train = train
        self.validation = validation
        self.test = test
        self.validation_support = validation_support
        self.validation_query = validation_query
        self.test_support = test_support
        self.test_query = test_query

class BundleFactory:
    """Create ``DataBundle`` objects from split data frames."""

    @staticmethod
    def _create_ds(df: pd.DataFrame, split: str, role: str = None) -> BaseDataset:
        """Return one ``BaseDataset`` for a split and optional support/query role.

        Model features exclude split, group, response, role, and true-effect
        bookkeeping columns. True fixed/random/noise parts are returned as
        optional aligned series for diagnostics when present in ``df``.
        """
        mask = (df[SPLIT_COLUMN_NAME] == split)
        
        if role:
            mask &= (df[CONTEXT_ROLE_COLUMN_NAME] == role)
        
        subset = df[mask].copy()

        # Special features that are not part of the model input and are dropped if they exist in the data frame.
        meta = {SPLIT_COLUMN_NAME, GROUPING_COLUMN_NAME, RESPONSE_COLUMN_NAME, CONTEXT_ROLE_COLUMN_NAME,
                FIXED_EFFECT_PART_NAME, RANDOM_EFFECT_PART_NAME, NOISE_PART_NAME}
        features = subset.drop(columns=list(meta & set(subset.columns)))

        get_column_if_exists = lambda col_name: subset[col_name].reset_index(drop=True) if col_name in subset.columns else None
        return BaseDataset(
            features=features.reset_index(drop=True),
            response=subset[RESPONSE_COLUMN_NAME].reset_index(drop=True),
            grouping_var=subset[GROUPING_COLUMN_NAME].reset_index(drop=True),
            # Include the true parts that formed the response (if available). Consumer needs to make sure that the model does not use it as a feature!
            fixed_effect_part=get_column_if_exists(FIXED_EFFECT_PART_NAME), 
            random_effect_part=get_column_if_exists(RANDOM_EFFECT_PART_NAME),
            noise_part=get_column_if_exists(NOISE_PART_NAME)
        )

    @classmethod
    def create_bundle(cls, df: pd.DataFrame, task: BaseTask) -> DataBundle:
        """Return the task-specific train/validation/test ``DataBundle``.

        In-context bundles contain ``train``, ``validation``, and ``test``.
        Few-shot bundles contain ``train`` plus support/query datasets for
        validation and test splits.
        """
        train_ds = cls._create_ds(df, TRAIN_SPLIT_NAME)
        
        if task.name == FEW_SHOT_TASK_NAME:
            return DataBundle(
                train=train_ds,
                validation_support=cls._create_ds(df, VALIDATION_SPLIT_NAME, role=SUPPORT_ROLE_NAME),
                validation_query=cls._create_ds(df, VALIDATION_SPLIT_NAME, role=TARGET_ROLE_NAME),
                test_support=cls._create_ds(df, TEST_SPLIT_NAME, role=SUPPORT_ROLE_NAME),
                test_query=cls._create_ds(df, TEST_SPLIT_NAME, role=TARGET_ROLE_NAME)
            )
        else:
            return DataBundle(
                train=train_ds,
                validation=cls._create_ds(df, VALIDATION_SPLIT_NAME),
                test=cls._create_ds(df, TEST_SPLIT_NAME)
            )
