import logging

from omegaconf import ListConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from typing import List

from src.constants import SPLIT_COLUMN_NAME, TRAIN_SPLIT_NAME, ENCODED_COLUMN_PREFIX

logger = logging.getLogger(__name__)

class DataTransform(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class LogTransformer(DataTransform):
    """Applies log1p transformation to specified columns.
    Args:
        columns: List of column names to apply log1p transformation to.
    """
    def __init__(self, columns: List):
        # Hydra passes it as a ListConfig, but we need a list for the df operations
        if isinstance(columns, ListConfig):
            self.columns = list(columns)
        else:
            self.columns = columns

    def __call__(self, df: pd.DataFrame):
        df = df.copy()
        for col in self.columns:
            df[col] = np.log1p(df[col])
        return df
    
    def __str__(self):
        return f"LogTransformer(columns={self.columns})"

class Standardizer(DataTransform):
    """Standardizes specified columns using training set statistics.
    Args:
        columns: List of column names to apply standardization to."""
    def __init__(self, columns: List):
        # Hydra passes it as a ListConfig, but we need a list for the df operations
        if isinstance(columns, ListConfig):
            self.columns = list(columns)
        else:
            self.columns = columns
        self.scaler = StandardScaler()

    def __call__(self, df: pd.DataFrame):
        logger.info(f"Numerical columns to standardize: {self.columns}. Will be done based on training set statistics.")
        df = df.copy()
        df_train = df[df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME]
        self.scaler.fit(df_train[self.columns])
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df
    
    def __str__(self):
        return f"Standardizer(columns={self.columns})"

class DummyEncoder(DataTransform):
    """Encodes categorical columns as dummy variables.
    Args:
        columns: List of column names to apply dummy encoding to."""
    def __init__(self, columns: List):
        # Hydra passes it as a ListConfig, but we need a list for the df operations
        if isinstance(columns, ListConfig):
            self.columns = list(columns)
        else:
            self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Categorical columns to encode: {self.columns}. Will be done based on training set categories.")
        df = df.copy()
    
        df_train = df[df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME]

        for col in self.columns:
            # Extract categories from training set only
            train_categories = sorted(pd.Series(df_train[col]).unique())
            
            # Values in the full df that are not in the training categories
            unseen = set(df[col].unique()) - set(train_categories)
            if unseen:
                logger.warning(f"Column '{col}' has values not present in training set: {unseen}."
                                f"These will be encoded as NaN.")
                
            # Make column categorical with training categories, to make column order consistent.
            df[col] = pd.Categorical(df[col], categories=train_categories)
        col_prefix = {col: f"{ENCODED_COLUMN_PREFIX}{col}" for col in self.columns}
        df = pd.get_dummies(df, columns=self.columns, prefix=col_prefix, drop_first=True, dtype=int)
        return df
    
    def __str__(self):
        return f"DummyEncoder(columns={self.columns})"
    
class MedianImputer(DataTransform):
    """Imputes missing values with the median of the training set.
    Args:
        columns: List of column names to apply median imputation to."""
    def __init__(self, columns: List):
        # Hydra passes it as a ListConfig, but we need a list for the df operations
        if isinstance(columns, ListConfig):
            self.columns = list(columns)
        else:
            self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Numerical columns to impute with median: {self.columns}. Will be done based on training set statistics.")
        df = df.copy()
        df_train = df[df[SPLIT_COLUMN_NAME] == TRAIN_SPLIT_NAME]
        for col in self.columns:
            median_value = df_train[col].median()
            df[col] = df[col].fillna(median_value)
        return df
    
    def __str__(self):
        return f"MedianImputer(columns={self.columns})"
    
class NACategoryImputer(DataTransform):
    """Imputes missing values in categorical columns with 'NA'.
    Args:
        columns: List of column names to apply NA imputation to."""
    def __init__(self, columns: List):
        # Hydra passes it as a ListConfig, but we need a list for the df operations
        if isinstance(columns, ListConfig):
            self.columns = list(columns)
        else:
            self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Categorical columns to impute with 'NA': {self.columns}.")
        df = df.copy()
        for col in self.columns:
            df[col] = df[col].fillna('NA')
        return df
    
    def __str__(self):
        return f"NACategoryImputer(columns={self.columns})"
