"""
Data providers and synthetic components for mixed-effects regression experiments.

This module defines two related pieces of the data pipeline:
- synthetic effect/noise components that construct responses as
  fixed_effect(x) + random_effect_g(x) + noise(x).
- real and synthetic data providers that write raw data and train-validation-test splits.
"""

# Third-party imports
from abc import ABC, abstractmethod
import os
import pandas as pd
import logging

import numpy as np
from typing import Sequence
from scipy.interpolate import RegularGridInterpolator
from typing import List, Union, Tuple, Dict, Any, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, Matern
import torch
from ucimlrepo import fetch_ucirepo

from src.task.base_task import BaseTask
from src.data.data_layout import SOURCE_DATA_FOLDER, RAW_DATA_FOLDER, SyntheticDataLayout, RealDataLayout

# Local imports
from ..constants import (GROUPING_COLUMN_NAME, RESPONSE_COLUMN_NAME, SPATIAL_COLUMN_PREFIX, 
SPATIAL_ID_COLUMN_NAME, FIXED_EFFECT_PART_NAME, RANDOM_EFFECT_PART_NAME, NOISE_PART_NAME, FIXED_ONLY_COLUMN_PREFIX)


logger = logging.getLogger(__name__)

class SyntheticDataComponent(ABC):
    """Named base class for configurable pieces of the synthetic generator."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def __str__(self):
        return self.name

class FixedEffect(SyntheticDataComponent):
    """Shared fixed-effect function"""

    @abstractmethod
    def generate(self, features: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def feature_dimension(self) -> int:
        pass

class RandomEffect(SyntheticDataComponent):
    """Group-specific stochastic function evaluated independently per group."""

    @abstractmethod
    def generate(self, features: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
        pass

class Noise(SyntheticDataComponent):
    """Observation-level noise."""

    @abstractmethod
    def generate(self, x: torch.Tensor, seed: int) -> np.ndarray:
        pass

class ZeroRandomEffect(RandomEffect):
    """Random-effect component that contributes no group-specific signal.
    
    Useful for plotting purposes when you want to visualize the fixed effect or the noise alone.
    """

    def generate(self, features: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
        return np.zeros(len(features))

class GPRandomEffect(RandomEffect):
    """Gaussian-process random effect with a kernel function."""

    def __init__(self, alpha: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    @abstractmethod
    def _create_kernel(self):
        pass

    def generate(self, features: np.ndarray, grouping_var: np.ndarray, seed: int) -> np.ndarray:
        random_effects = np.zeros(len(features))
        kernel = self._create_kernel()
        
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            optimizer=None
        )

        unique_groups = np.unique(grouping_var)
        
        rng = np.random.default_rng(seed)
        group_seeds = rng.integers(0, 2**31, size=len(unique_groups))

        for i, group_id in enumerate(unique_groups):
            group_mask = (grouping_var == group_id)
            group_x = features[group_mask]

            group_random_effect = gpr.sample_y(
                group_x, 
                random_state=int(group_seeds[i])
            ).flatten()

            random_effects[group_mask] = group_random_effect

        return random_effects
    
class RBFGPRandomEffect(GPRandomEffect):
    """Gaussian-process random effect with an RBF covariance kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.length_scale = length_scale
        self.variance = variance

    def _create_kernel(self):

        return self.variance * RBF(length_scale=self.length_scale)

class MaternGPRandomEffect(GPRandomEffect):
    """Gaussian-process random effect with a Matern covariance kernel."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.length_scale = length_scale
        self.variance = variance
        self.nu = nu

    def _create_kernel(self):
        return self.variance * Matern(length_scale=self.length_scale, nu=self.nu)


# Khoshnevisan, D. (2002). Multiparameter processes: An introduction to
# random fields. Springer. Ch. 5, Sec. 1.5 defines the N-parameter,
# one-dimensional Brownian sheet with covariance prod_i min(s_i, t_i).
# The N = 1 special case is Brownian motion.
class BrownianMotionKernel(Kernel):
    """Non-stationary Brownian-motion covariance kernel.

    For each feature dimension, covariance is proportional to min(x, y) after
    shifting by ``lower_bound``.
    Multi-dimensional inputs use the Brownian-sheet product covariance.
    """

    def __init__(self, variance=1.0, lower_bound=0.0):
        self.variance = variance
        self.lower_bound = lower_bound

    def _shift(self, X):
        X = np.atleast_2d(X).astype(float)
        lb = np.asarray(self.lower_bound, dtype=float)
        if lb.ndim == 0:
            lb = np.full(X.shape[1], lb)
        return X - lb

    def __call__(self, X, Y=None, eval_gradient=False):
        Xs = self._shift(X)
        Ys = Xs if Y is None else self._shift(Y)

        if np.any(Xs < 0) or np.any(Ys < 0):
            raise ValueError("Inputs must be >= lower_bound coordinate-wise")

        K = np.ones((Xs.shape[0], Ys.shape[0]))
        for d in range(Xs.shape[1]):
            K *= np.minimum.outer(Xs[:, d], Ys[:, d])
        K *= self.variance

        if eval_gradient:
            grad = (K / self.variance)[..., np.newaxis]
            return K, grad

        return K

    def diag(self, X):
        Xs = self._shift(X)
        if np.any(Xs < 0):
            raise ValueError("Inputs must be >= lower_bound coordinate-wise")
        return self.variance * np.prod(Xs, axis=1)

    # The kernel is not stationary, since it depends on the absolute values of the inputs.
    # A stationary kernel only depends on the difference, i.e. k(x,y) = k*(x-y) for some function k*.
    def is_stationary(self):
        return False

class BrownianMotionRandomEffect(GPRandomEffect):
    """Gaussian-process random effect using the Brownian-motion kernel."""

    def __init__(self, variance: float = 1.0, lower_bound=0.0, **kwargs):
        super().__init__(**kwargs)
        self.variance = variance
        self.lower_bound = lower_bound

    def _create_kernel(self):
        return BrownianMotionKernel(
            variance=self.variance,
            lower_bound=self.lower_bound,
        )

class HomoscedasticGaussianNoise(Noise):
    """Gaussian observation noise with constant standard deviation."""

    def __init__(self, std: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    def generate(self, features: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=features.shape[0]) * self.std
        return noise

class LogNormalNoise(Noise):
    """Centered log-normal noise which can be scaled to a target variance."""

    def __init__(self, sigma: float = 1.0, variance: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self.sigma = sigma

        if variance is not None:
            if variance <= 0:
                raise ValueError("Variance must be positive")
            # If variance is provided, mu is set such that the resulting log-normal distribution has the specified variance.
            self.variance = variance
            self._mu = 0.5 * np.log(variance / (np.exp(sigma ** 2) - 1.0)) - 0.5 * sigma ** 2
        else:
            # Otherwise, we use a mu of 0.
            self._mu = 0.0
            self.variance = (np.exp(sigma ** 2) - 1.0) * np.exp(sigma ** 2)

        # Calculate the offset to ensure zero mean. The mean of a log-normal distribution is exp(mu + sigma^2 / 2).
        self._offset = np.exp(self._mu + sigma ** 2 / 2.0)
        logger.info(f"LogNormalNoise: sigma={sigma:.4f}, variance={self.variance:.4f}, internal mean={self._mu:.4f}")

    def generate(self, features: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.lognormal(mean=self._mu, sigma=self.sigma, size=features.shape[0]) - self._offset

class StepFixedEffect(FixedEffect):
    """
    Piecewise-constant fixed effect.

    Accepted input forms:
    - 1D:
        breaks=[0.0]
        values=[-5.0, 5.0]
    - 2D:
        breaks=[[0.0], [1.5, 3.0]] # breaks x dim at 0.0, breaks in y dim at 1.5 and 3.0
        values=[[...], [...]] # values of the step function in each grid cell

    A multiplier can be provided to scale the all values of the step function.
    """

    def __init__(self, breaks, values, multiplier: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)

        self.breaks = self._normalize_breaks(breaks)
        self.values = np.asarray(values, dtype=float)

        self.multiplier = multiplier
        
        # If there are n breaks in a dimension, there are n+1 grid cells in that dimension, 
        # and thus n+1 values in that dimension that need to be provided.
        expected_shape = tuple(len(b) + 1 for b in self.breaks)
        if self.values.shape != expected_shape:
            raise ValueError(f"values.shape must be {expected_shape}, got {self.values.shape}")

    @staticmethod
    def _normalize_breaks(breaks) -> list[np.ndarray]:
        if not isinstance(breaks, Sequence):
            raise TypeError("breaks must be a sequence")

        if len(breaks) == 0:
            raise ValueError("breaks must not be empty")

        first = breaks[0]

        # 1D case: e.g. breaks=[0.0, 1.0]
        if np.isscalar(first):
            arr = np.asarray(breaks, dtype=float)
            return [arr]

        # multi-D case: e.g. breaks=[[0.0], [1.0, 2.0]]
        out = [np.atleast_1d(np.asarray(b, dtype=float)) for b in breaks]
        return out


    def generate(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(features)

        if features.shape[1] != len(self.breaks):
            raise ValueError(f"Expected {len(self.breaks)} feature dimensions, got {features.shape[1]}")

        # Map continuous values to bin coordinates
        grid_coords = []
        for dim, cuts in enumerate(self.breaks):
            bin_idx = np.searchsorted(cuts, features[:, dim], side="right")
            grid_coords.append(bin_idx)

        # In 2D, matrix Row 0 is the top (High Y), while bin_idx 0 is the bottom (Low Y)
        # We need to flip it.
        if len(self.breaks) == 2:
            x_coords, y_coords = grid_coords[0], grid_coords[1]
            
            # Invert the Y coordinates to map "High Y" to "Row 0"
            num_rows = self.values.shape[0]
            row_coords = (num_rows - 1) - y_coords
            col_coords = x_coords
            
            lookup_indices = (row_coords, col_coords)
        else:
            # For all other dimensions, we use the bin indices as they are.
            lookup_indices = tuple(grid_coords)

        return self.multiplier * self.values[lookup_indices]

    @property
    def feature_dimension(self) -> int:
        return len(self.breaks)


class LinearFixedEffect(FixedEffect):
    """Linear fixed effect ``features @ slopes + intercept``.

    Args:
        slopes: array of slopes for each feature dimension.
        intercept: scalar intercept term.
    """

    def __init__(self, slopes: list, intercept: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.slopes = np.asarray(slopes)
        self.intercept = intercept

    def generate(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] != self.feature_dimension:
            raise ValueError(f"Feature dimension {features.shape[1]} does not match number of slopes {self.feature_dimension}")
        return features @ self.slopes + self.intercept
    
    @property
    def feature_dimension(self) -> int:
        return len(self.slopes)



class DataProvider(ABC):
    """Base interface for synthetic and real data providers that create raw and train-validation-test datasets."""

    def __init__(self, experiment_name: str, transforms: List = None):
        self.experiment_name = experiment_name
        self.transforms = transforms or []

    @abstractmethod
    def generate_raw_data(self) -> None:
        pass
    
    @abstractmethod
    def generate_split_data(self) -> None:
        pass

    def _apply_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        for transform in self.transforms:
            logger.info(f"Applying transform: {transform}")
            df = transform(df)
        return df

class RealDataProvider(DataProvider):
    """Base provider for real datasets stored in the shared experiment format.

    Subclasses implement ``_load_dataset`` and must return a DataFrame with
    feature columns plus ``GROUPING_COLUMN_NAME`` and ``RESPONSE_COLUMN_NAME``.
    This base class stores the raw data, generates the train-validation-test splits, and applies optional
    transforms (e.g. scaling, one-hot encoding).

    Args:
        experiment_name: Name of the experiment, used to create a folder for storing raw and split data.
        transforms: List of transforms to apply after splitting.
        response_column_names: List of response column names to create a dataset for. (mainly for legacy reasons, since Parkinsons data had two response columns)
        spatial_columns: Optional list of columns that are spatial coordinates, which will be renamed with a prefix. (currently unused)
        spatial_id: Optional column name that will be handled as a spatial ID, which will be renamed with a prefix. (currently unused)
        min_group_size: Minimum number of observations per group to keep in the dataset.
        random_effect_drop_columns: Optional list of columns that should be marked as fixed-effect-only, which will not be passed as random effect features. 
    """

    def __init__(
        self,
        experiment_name,
        transforms,
        response_column_names,
        split_seed_list,
        spatial_columns=None,
        spatial_id=None,
        min_group_size=15,
        random_effect_drop_columns=None,
    ):
        super().__init__(experiment_name, transforms)
        self.response_column_names = response_column_names
        self.split_seed_list = split_seed_list
        self.spatial_columns = spatial_columns
        self.spatial_id = spatial_id
        self.min_group_size = min_group_size
        self.random_effect_drop_columns = list(random_effect_drop_columns or [])

    def generate_raw_data(self):
        """Load each configured response dataset and write its raw parquet file."""

        for response in self.response_column_names:
            df = self._load_dataset(response)
            path = RealDataLayout.get_raw_df_path(self.experiment_name, response)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path, index=False)

    def generate_split_data(self, tasks: List[BaseTask]):
        """Create task-specific split parquet files for every seed and response."""

        for seed in self.split_seed_list:
            for response in self.response_column_names:
                raw_path = RealDataLayout.get_raw_df_path(self.experiment_name, response)
                raw_df = pd.read_parquet(raw_path)
                for task in tasks:
                    split_df = task.split_data(raw_df, seed)
                    split_df = self._apply_transforms(split_df)
                    split_df = self._mark_fixed_only_columns(split_df)
                    if self.spatial_columns:
                        logger.info(f"Renaming spatial columns with prefix '{SPATIAL_COLUMN_PREFIX}'.")
                        split_df.rename(columns={col: f"{SPATIAL_COLUMN_PREFIX}{col}" for col in self.spatial_columns}, inplace=True)
                    if self.spatial_id:
                        logger.info(f"Renaming spatial ID column with name '{SPATIAL_ID_COLUMN_NAME}'.")
                        split_df.rename(columns={self.spatial_id: f"{SPATIAL_ID_COLUMN_NAME}"}, inplace=True)
                    
                    out_path = RealDataLayout.get_split_df_path(task, self.experiment_name, response, seed)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    split_df.to_parquet(out_path, index=False)

    def _drop_low_group_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop groups with fewer than ``self.min_group_size`` observations."""
        logger.info(f"Only keeping groups with at least {self.min_group_size} observations.")
        counts_group = df[GROUPING_COLUMN_NAME].value_counts()
        n_dropped_groups = (counts_group < self.min_group_size).sum()
        n_total_groups = len(counts_group)
        logger.info(f"Number of groups kept: {n_total_groups - n_dropped_groups} ({(n_total_groups - n_dropped_groups) / n_total_groups:.1%})")
        n_dropped_observations = counts_group[counts_group < self.min_group_size].sum()
        n_total_observations = len(df)
        logger.info(f"Number of observations kept: {n_total_observations - n_dropped_observations} ({(n_total_observations - n_dropped_observations) / n_total_observations:.1%})")
        return df[df[GROUPING_COLUMN_NAME].map(counts_group) >= self.min_group_size]

    def _mark_fixed_only_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prefix configured columns so random-effect models can ignore them."""

        if not self.random_effect_drop_columns:
            return df

        rename_map = {}
        for column in self.random_effect_drop_columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' configured as random-effect drop column but not found.")
                continue
            if str(column).startswith(FIXED_ONLY_COLUMN_PREFIX):
                continue
            rename_map[column] = f"{FIXED_ONLY_COLUMN_PREFIX}{column}"

        if rename_map:
            logger.info(
                f"Marking columns as fixed-effect-only with prefix '{FIXED_ONLY_COLUMN_PREFIX}': "
                f"{list(rename_map)}"
            )
            df = df.rename(columns=rename_map)
        return df

    def fetch_raw_df(self, response_column: str) -> pd.DataFrame:
        """Load a previously generated raw real-data DataFrame."""

        path = RealDataLayout.get_raw_df_path(self.experiment_name, response_column)
        print(f"Logger name: {logger}")
        logger.info(f"Fetching raw real data from {path}")
        return pd.read_parquet(path)

    def fetch_split_df(self, task: BaseTask, response_column: str, seed: int) -> pd.DataFrame:
        """Load a previously generated real-data split for one task and seed."""

        path = RealDataLayout.get_split_df_path(task, self.experiment_name, response_column, seed)
        logger.info(f"Fetching split real data from {path}")
        return pd.read_parquet(path)
    
    @abstractmethod
    def _load_dataset(self, response_name: str) -> pd.DataFrame:
        pass

class CarsDataProvider(RealDataProvider):
    """Provider for used-car listings with car model as the grouping variable.

    We follow closely the preprocessing steps from Simchoni and Rosset (2023), "Integrating Random Effects
    in Deep Neural Networks", JMLR 24(156):1-57.
    Paper: https://jmlr.org/papers/v24/22-0501.html
    Code:  https://github.com/gsimchoni/lmmnn/blob/main/r_scripts/ETL/cars_etl.R
           (last accessed: 2026-08-08)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _load_dataset(self, response_name: str) -> pd.DataFrame:
        data_path = SOURCE_DATA_FOLDER / "vehicles.csv"
        df = pd.read_csv(data_path)
        
        features = ["price", "manufacturer", "model", "year", "condition", "odometer",
            "transmission", "drive", "lat", "long", "VIN"]
        df = df[features]

        initial_length = len(df)
        previous_length = initial_length
        logger.info(f"Initial number of rows: {initial_length}")

        # VIN = Vehicle Identification Number, which should be unique for each car.
        df = df.drop_duplicates(subset="VIN", keep="first")
        current_length = len(df)
        logger.info(f"Number of rows after VIN deduplication: {current_length} (removed {previous_length - current_length} duplicates)")
        previous_length = current_length
        df.drop(columns=["VIN"], inplace=True)

        df = df.dropna(subset=["year", "price", "lat", "long"])
        current_length = len(df)
        logger.info(f"Number of rows after dropping NAs for price, year, lat, and long: {current_length} (removed {previous_length - current_length} rows)")
        previous_length = current_length

        df = df[
            df["price"].between(1000, 300000) & 
            df["lat"].between(20, 50) & 
            df["long"].between(-150, -50)
        ]
        current_length = len(df)
        logger.info(f"Number of rows after filtering unrealistic values for price, lat, and long: {current_length} (removed {previous_length - current_length} rows)")
        previous_length = current_length

        top10_manufacturers = df["manufacturer"].value_counts().nlargest(10).index
        df["manufacturer"] = np.where(df["manufacturer"].isin(top10_manufacturers), 
                                    df["manufacturer"], "other")

        df["model_id"] = pd.factorize(df["model"])[0]

        df.drop(columns=["model"], inplace=True)

        df["log_price"] = np.log(df["price"])
        df.drop(columns=["price"], inplace=True)

        df.rename(columns={"model_id": GROUPING_COLUMN_NAME, "log_price": RESPONSE_COLUMN_NAME}, inplace=True)
        df = self._drop_low_group_size(df)
        return df

class SpotifyDataProvider(RealDataProvider):
    """Provider for Spotify songs data with artist as the grouping variable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _load_dataset(self, response_name: str) -> pd.DataFrame:
        data_path = SOURCE_DATA_FOLDER /"spotify_songs.csv"
        df = pd.read_csv(data_path)
        previous_length = len(df)
        logger.info(f"Initial number of rows: {previous_length}")
        df.drop_duplicates(subset=["track_id"], inplace=True)
        current_length = len(df)
        logger.info(f"Number of rows after track ID deduplication: {current_length} (removed {previous_length - current_length} duplicates)")

        df["artist_id"] = pd.factorize(df["track_artist"])[0]
        df.rename(columns={"artist_id": GROUPING_COLUMN_NAME, response_name: RESPONSE_COLUMN_NAME}, inplace=True)
        
        numerical_features = ["energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
        categorical_features = ["playlist_genre", "key", "mode"]
        df = df[numerical_features + categorical_features + [GROUPING_COLUMN_NAME, RESPONSE_COLUMN_NAME]]
        df = self._drop_low_group_size(df)
        return df

class BikeDataProvider(RealDataProvider):
    """Provider for Seoul bike rentals with day as the grouping variable."""

    UCI_DATASET_ID = 560
    SOURCE_COPY_FILENAME = "SeoulBikeData.csv"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _source_copy_path(self):
        return SOURCE_DATA_FOLDER / self.SOURCE_COPY_FILENAME

    def _fetch_source_dataset(self) -> pd.DataFrame:
        source_copy_path = self._source_copy_path()
        logger.info(f"Fetching Seoul Bike dataset from UCI ML Repository (id={self.UCI_DATASET_ID}).")
        dataset = fetch_ucirepo(id=self.UCI_DATASET_ID)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

        os.makedirs(source_copy_path.parent, exist_ok=True)
        df.to_csv(source_copy_path, index=False)
        logger.info(f"Stored source copy at {source_copy_path}.")
        return df
    
    def _load_dataset(self, response_name: str) -> pd.DataFrame:
        df = self._fetch_source_dataset()
        previous_length = len(df)
        logger.info(f"Initial number of rows: {previous_length}")

        df = df[df["Functioning Day"] == "Yes"].copy()
        current_length = len(df)
        logger.info(f"Number of rows after keeping functioning days: {current_length} (removed {previous_length - current_length} rows)")

        df.rename(
            columns={
                "Rented Bike Count": "rented_bike_count",
                "Hour": "hour",
                "Temperature": "temperature",
                "Humidity": "humidity",
                "Rainfall": "rainfall",
                "Seasons": "season",
            },
            inplace=True,
        )

        df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values(["date", "hour"]).reset_index(drop=True)

        first_date = df["date"].min()
        df["date_id"] = (df["date"] - first_date).dt.days.astype(int)
        df["day_of_week"] = df["date"].dt.dayofweek.astype(float)
        df["day_of_year"] = df["date"].dt.dayofyear.astype(float)

        # Encode cyclical features with sine and cosine transformations, so that
        # neighbouring values stay close: otherwise Sunday (weekday 6) would look far
        # from Monday (weekday 0), and 31 Dec far from 1 Jan.
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["log_count"] = np.log1p(df["rented_bike_count"])
        df.rename(columns={"date_id": GROUPING_COLUMN_NAME, "log_count": RESPONSE_COLUMN_NAME}, inplace=True)

        features = [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "temperature",
            "humidity",
            "rainfall",
        ]
        df = df[features + [GROUPING_COLUMN_NAME, RESPONSE_COLUMN_NAME]]
        df = self._drop_low_group_size(df)
        return df

class CowsDataProvider(RealDataProvider):
    """Provider for the nlme Milk dataset with cow as the grouping variable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_dataset(self, response_name: str) -> pd.DataFrame:
        df = pd.read_csv(SOURCE_DATA_FOLDER / "nlme_Milk.csv")
        previous_length = len(df)
        logger.info(f"Initial number of rows: {previous_length}")

        df = df.drop(columns=["rownames"], errors="ignore")
        df = df.dropna(subset=["protein", "Time", "Cow", "Diet"])
        current_length = len(df)
        logger.info(f"Number of rows after dropping NAs: {current_length} (removed {previous_length - current_length} rows)")

        df.rename(
            columns={
                "protein": RESPONSE_COLUMN_NAME,
                "Cow": GROUPING_COLUMN_NAME,
                "Time": "time",
                "Diet": "diet",
            },
            inplace=True,
        )

        df[GROUPING_COLUMN_NAME] = pd.factorize(df[GROUPING_COLUMN_NAME])[0]
        df["diet"] = df["diet"].astype(str)
        df = df[["time", "diet", GROUPING_COLUMN_NAME, RESPONSE_COLUMN_NAME]]
        df = self._drop_low_group_size(df)
        return df

class SyntheticDataProvider(DataProvider):
    """Generate synthetic grouped regression data from configured components.

    For each seed, rows are generated for every subject and response values are
    decomposed into fixed-effect, random-effect, and noise columns. 
    Relevant features are passed to all configured components; 
    optional irrelevant features are stored as predictors but do not affect the generated response.

    Inputs:
    - n_subjects: Number of subjects (groups) to generate.
    - n_observations_per_subject: Number of observations per subject.
    - features_domain: List of floats specifying the range of each feature dimension.
    - fixed_effect: Fixed effect class to generate the shared fixed effects.
    - random_effect: Random effect class to generate random effects for each subject.
    - noise: Noise class to generate noise values.
    """

    def __init__(self, n_subjects: int, n_observations_per_subject: int, 
                 features_domain: List[float], fixed_effect: FixedEffect, 
                 random_effect: RandomEffect, noise: Noise, seed_list: List[int], 
                 n_irrelevant_features: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.n_subjects = n_subjects
        self.n_observations_per_subject = n_observations_per_subject
        self.features_domain = features_domain
        self.fixed_effect = fixed_effect
        self.random_effect = random_effect
        self.noise = noise
        self.seed_list = seed_list
        self.n_irrelevant_features = n_irrelevant_features
        logger.info(f"Generating synthetic data ...")
        logger.info(f"Number of subjects: {self.n_subjects}")
        logger.info(f"Number of observations per subject: {self.n_observations_per_subject}")
        logger.info(f"Features domain: {self.features_domain}")
        logger.info(f"Fixed effect: {self.fixed_effect}")
        logger.info(f"Random effect: {self.random_effect}")
        logger.info(f"Noise: {self.noise}")
        logger.info(f"Seed list: {self.seed_list}")
        self.raw_df_dict = {}
        self.split_df_dict = {}

    def generate_raw_data(self, save_data: bool = True):
        """Generate one synthetic dataset per seed.

        The response is additive: y = fixed_effect + random_effect + noise.
        The optional irrelevant features are generated but do not affect the response.

        Returns a {seed: df} mapping (also stored in ``self.raw_df_dict``), with columns
        for the group, the response, the three ground-truth components (can be used for debugging / plotting purposes), and the relevant/irrelevant features. 
        Optionally writes each df to Parquet.
        """

        seed_df_dict = {} # key: seed, value: generated df for that seed
        for seed in self.seed_list:
            logger.info(f"Generating synthetic data with seed {seed}...")
            rng = np.random.default_rng(seed=seed)
            features_seed, random_effects_seed, noise_seed = rng.integers(0, 10**6, size=3)
            
            groups = np.repeat(np.arange(self.n_subjects), self.n_observations_per_subject)
            if self.n_irrelevant_features > 0:
                logger.info(f"Generating {self.fixed_effect.feature_dimension} relevant features (used for fixed effect and random effect).")
                logger.info(f"Generating {self.n_irrelevant_features} irrelevant features.")
            full_features = self._generate_features(features_seed)
            relevant_features = full_features[:, :self.fixed_effect.feature_dimension]
            irrelevant_features = full_features[:, self.fixed_effect.feature_dimension:]
            logger.info("Generating fixed effects...")
            fixed_effects = self.fixed_effect.generate(relevant_features)
            logger.info("Generating random effects...")
            random_effects = self.random_effect.generate(relevant_features, groups, seed=random_effects_seed)
            logger.info(f"Generating noise...")
            noise = self.noise.generate(relevant_features, seed=noise_seed)
            
            y = fixed_effects + random_effects + noise

            df = pd.DataFrame({
                GROUPING_COLUMN_NAME: groups,
                RESPONSE_COLUMN_NAME: y,
                FIXED_EFFECT_PART_NAME: fixed_effects,
                RANDOM_EFFECT_PART_NAME: random_effects,
                NOISE_PART_NAME: noise
            })
            for i in range(self.fixed_effect.feature_dimension):
                df[f"feature_{i}"] = relevant_features[:, i]
            for i in range(self.n_irrelevant_features):
                df[f"irrelevant_feature_{i}"] = irrelevant_features[:, i]
            seed_df_dict[seed] = df

            logger.info(f"Synthetic data generated successfully.")
            if save_data:
                raw_df_path = SyntheticDataLayout.get_raw_df_path(self.experiment_name, self.fixed_effect, seed=seed)
                logger.info(f"Saving synthetic data into {raw_df_path}")
            
                os.makedirs(os.path.dirname(raw_df_path), exist_ok=True)
                df.to_parquet(raw_df_path, index=False)
            else: 
                logger.info(f"Not saving synthetic data.")
        self.raw_df_dict = seed_df_dict
        return self.raw_df_dict
    
    def generate_split_data(self, tasks: List[BaseTask], save_data: bool = True):
        """Generate train-validation-test splits from raw data for all seeds and based on prediction task (in-context / few-shot).
        ``generate_raw_data()`` must be called first so that the raw data is available.
        """

        seed_df_dict = {} # key: (task, seed) -> value: generated df for that seed
        for seed in self.seed_list:
            raw_df = self.raw_df_dict[seed]
            for task in tasks:
                split_df = task.split_data(raw_df, seed)
                self._apply_transforms(split_df)
                seed_df_dict[(task.name, seed)] = split_df
                if save_data:
                    split_df_path = SyntheticDataLayout.get_split_df_path(task, self.experiment_name, self.fixed_effect, seed)
                    os.makedirs(os.path.dirname(split_df_path), exist_ok=True)
                    split_df.to_parquet(split_df_path, index=False)
        self.split_df_dict = seed_df_dict
        return self.split_df_dict


    def _generate_features(self, seed):
        rng = np.random.default_rng(seed)
        feature_dimension = self.fixed_effect.feature_dimension
        noise_feature_dimension = self.n_irrelevant_features
        return rng.uniform(
            low=self.features_domain[0],
            high=self.features_domain[1],
            size=(self.n_subjects * self.n_observations_per_subject, feature_dimension + noise_feature_dimension),
        )

    def _save_raw_df(self, df, seed):
        raw_df_path = SyntheticDataLayout.get_raw_df_path(
            experiment_name=self.experiment_name,
            fixed_effect=self.fixed_effect,
            seed=seed
        )
        os.makedirs(os.path.dirname(raw_df_path), exist_ok=True)
        df.to_parquet(raw_df_path, index=False)

    def _save_split_df(self, split_df, task, seed):
        split_df_path = SyntheticDataLayout.get_split_df_path(
                task=task,
                experiment_name=self.experiment_name,
                fixed_effect=self.fixed_effect,
                seed=seed
            )
        os.makedirs(os.path.dirname(split_df_path), exist_ok=True)
        split_df.to_parquet(split_df_path, index=False)


    def fetch_raw_df(self, seed: int) -> pd.DataFrame:
        """Load previously generated raw data."""

        path = SyntheticDataLayout.get_raw_df_path(self.experiment_name, self.fixed_effect, seed)
        logger.info(f"Fetching raw synthetic data (Seed {seed}) from {path}")
        return pd.read_parquet(path)

    def fetch_split_df(self, task: Any, seed: int) -> pd.DataFrame:
        """Load a previously generated synthetic split for one task and seed."""

        path = SyntheticDataLayout.get_split_df_path(task, self.experiment_name, self.fixed_effect, seed)
        logger.info(f"Fetching split synthetic data (Task {task.name}, Seed {seed}) from {path}")
        return pd.read_parquet(path)
