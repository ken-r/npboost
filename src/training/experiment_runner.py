"""This module defines the classes responsible for running the experiments.

The ExperimentRunner class defines the common interface that the experiments need to implement.
Then all the different experiments (NPBoost, NP, Gradient-boosted trees, GPLinear, LME and TabICL) are implemented.
"""
from abc import abstractmethod, ABC
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import joblib
from matplotlib import pyplot as plt
import numpy as np
import gpboost as gpb
from gpboost import GPModel
from gpboost import Dataset as GPBDataset
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.optim import Adam
from lightgbm import Booster, Dataset, train as lgb_train
from lightgbm.callback import EarlyStopException as LightGBMEarlyStopException
from tabicl import TabICLRegressor
import wandb

from src.data.data_layout import PROJECT_ROOT
from src.task.in_context import InContext
from src.plot.model_diagnostics import plot_fixed_effect, plot_residuals_vs_predicted_single, plot_single_model_coverage, plot_single_model_predictions, plot_variance_components, plot_fixed_effect_density, plot_fixed_effect_vs_response
from src.constants import (
    EVALUATION_SEED_OFFSETS,
    FEW_SHOT_TASK_NAME,
    GROUPING_COLUMN_NAME,
    IN_CONTEXT_TASK_NAME,
    PREDICTION_SEED_OFFSETS,
    TEST_SPLIT_NAME,
    VALIDATION_SPLIT_NAME,
)
from src.data.datasets import (
    BaseDataset,
    DataBundle,
    GradientSplitDataset,
    ModelDataAdapter,
    NPBDataset,
    PairedFunctionSampler,
    PairedNPBDataset,
    gradient_split_collate_fn,
    paired_collate_fn,
)
from src.evaluation.crps import crps_eval, gaussian_crps, quantile_crps
from src.evaluation.metrics import groupwise_metrics
from src.evaluation.squared_error import point_pred_eval
from src.task.base_task import BaseTask
from src.training.losses import NPMLLoss
from src.utils.helpers import set_seed
from src.models.neural_process import NeuralProcess
from src.models.npboost import NPBoost
from src.training.np_trainer import NPTrainer
from src.evaluation.model_predictions import ModelPrediction, GroupedModelPrediction

logger = logging.getLogger(__name__)

INTERMEDIATE_PLOTTING_ROUNDS = [1, 2, 5, 10, 25, 50, 100, 150, 250]
FIXED_EFFECT_DIAGNOSTIC_MODEL_NAMES = {
    "npboost",
    "anpboost",
}


def _supports_fixed_effect_diagnostics(
    model_name: str,
    predictions: ModelPrediction | GroupedModelPrediction,
) -> bool:
    """Whether fixed-effect diagnostic plots are meaningful for this model."""
    return (
        str(model_name).lower() in FIXED_EFFECT_DIAGNOSTIC_MODEL_NAMES
        and predictions.fixed_effect is not None
    )


def _combined_true_effect_lines(
    context_data: Optional[BaseDataset],
    target_data: BaseDataset,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Return true-effect lines aligned as context rows followed by target rows."""
    if (
        context_data is None
        or context_data.fixed_effect_part is None
        or target_data.fixed_effect_part is None
    ):
        return None, None

    true_fixed_effect = pd.concat(
        [context_data.fixed_effect_part, target_data.fixed_effect_part],
        ignore_index=True,
    )

    if context_data.random_effect_part is None or target_data.random_effect_part is None:
        return true_fixed_effect, None

    true_random_effect = pd.concat(
        [context_data.random_effect_part, target_data.random_effect_part],
        ignore_index=True,
    )
    return true_fixed_effect, true_random_effect


@contextmanager
def force_torch_cpu_load():
    """A context manager to temporarily force PyTorch loading on CPU.

    This is useful when loading joblib-serialized PyTorch models on environments
    without CUDA support, ensuring the global torch.load state is safely restored
    even if errors occur during unpickling.
    """
    original_torch_load = torch.load

    def forced_cpu_load(*args, **kwargs):
        kwargs['map_location'] = 'cpu'
        return original_torch_load(*args, **kwargs)

    torch.load = forced_cpu_load
    try:
        yield
    finally:
        torch.load = original_torch_load


class ExperimentRunner(ABC):
    """Shared base class for all experiment runners

    Subclasses implement the model-specific data conversion and model calls in
    ``fit`()` and ``predict()``. 
    The ``run()`` method performs the full experiment (fitting, predicting, evaluating).  

    The ``predict()`` function returns predictions aligned row-for-row with ``target_data.response``.  
    The ``evaluate()`` function returns a split metric dictionary, usually containing ``crps`` and ``rmse`` metrics.
    The ``save_model()`` function saves a model whereas
    The ``load_model()`` function loads a saved model.

    Args:
        name: Name of the experiment/model.
        experiment_seed: Random seed for reproducibility.
        task: Prediction task (scenario) for which the model is applied.
        validation_metric: Metric used for early stopping.
        plot_intermediate: Whether to plot intermediate diagnostics during training (if supported by the model).
    """

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, validation_metric: str, plot_intermediate: bool = False):
        self.name = name
        self.experiment_seed = experiment_seed
        self.task = task
        self.validation_metric = validation_metric
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plot_intermediate = plot_intermediate

        set_seed(experiment_seed)
        self.is_fitted = False
        self.is_gaussian_prediction = True
        self.device = torch.device("cpu")
        self.crps_samples = 20

    def _prediction_seed(self, split_name: str) -> int:
        """Deterministic prediction seed for one evaluation split."""
        return self.experiment_seed + PREDICTION_SEED_OFFSETS[split_name]

    def _evaluation_seed(self, split_name: str) -> int:
        """Deterministic metric-evaluation seed for one split."""
        return self.experiment_seed + EVALUATION_SEED_OFFSETS[split_name]

    def _set_prediction_seed(self, split_name: str) -> None:
        set_seed(self._prediction_seed(split_name))

    def _set_evaluation_seed(self, split_name: str) -> None:
        set_seed(self._evaluation_seed(split_name))

    def run(self, data_bundle: DataBundle) -> Dict[str, Dict[str, float]]:
        """Run the standard fit-predict-evaluate pipeline.

        Returns:
            A mapping from split name (``VALIDATION_SPLIT_NAME`` and ``TEST_SPLIT_NAME``) to corresponding dictionary of metrics.
        """
        results = {}
        train_data = data_bundle.train
        validation_context, validation_target, test_context, test_target = self._resolve_splits(data_bundle)

        self.fit(train_data, validation_target_data=validation_target, validation_context_data=validation_context)
        # Fix the final prediction and evaluation seeds.
        # This allows us to generate exactly the same predictions again (if we save and load this model).
        self._set_prediction_seed(VALIDATION_SPLIT_NAME)
        validation_predictions = self.predict(validation_target, context_data=validation_context)
        self._set_prediction_seed(TEST_SPLIT_NAME)
        test_predictions = self.predict(test_target, context_data=test_context)

        self._set_evaluation_seed(VALIDATION_SPLIT_NAME)
        results[VALIDATION_SPLIT_NAME] = self.evaluate(validation_predictions, validation_target)
        self._set_evaluation_seed(TEST_SPLIT_NAME)
        results[TEST_SPLIT_NAME] = self.evaluate(test_predictions, test_target)
        return results

    def _resolve_splits(self, data_bundle: DataBundle) -> Tuple[BaseDataset, BaseDataset, BaseDataset, BaseDataset]:
        """Resolve validation/test context and target datasets for the current task."""
        if self.task.name == IN_CONTEXT_TASK_NAME:
            return (
                data_bundle.train,        # validation_context
                data_bundle.validation,   # validation_target
                data_bundle.train,        # test_context
                data_bundle.test,         # test_target
            )
        elif self.task.name == FEW_SHOT_TASK_NAME:
            return (
                data_bundle.validation_support, # validation context
                data_bundle.validation_query,   # validation target
                data_bundle.test_support,       # test context
                data_bundle.test_query,         # test target
            )
        else:
            raise ValueError(f"Unknown task name: {self.task.name}")

    @abstractmethod
    def fit(self, train_data: BaseDataset, validation_target_data: BaseDataset, validation_context_data: Optional[BaseDataset] = None):
        """Fit the model on training data.

        Args:
            train_data: Training data for model fitting.
            validation_target_data: Validation rows used for early stopping or model selection.
            validation_context_data: Optional validation context rows. For few-shot validation, some context points for the new task must be provided.
                For NP-based models, this context data must be provided.
        """
        pass

    @abstractmethod
    def predict(self, target_data: BaseDataset, context_data: Optional[BaseDataset] = None) -> ModelPrediction:
        """Predict target rows, optionally conditioned on split-specific context.

        Returns:
            A ``ModelPrediction`` or subclass whose arrays are aligned to
            ``target_data`` row order.
        """
        pass

    def evaluate(self, predictions: ModelPrediction, target_data: BaseDataset) -> Dict[str, float]:
        """Evaluate predictions against target responses.

        Gaussian predictions are analytically evaluated. 
        NP-based models are evaluated using sample-based methods.
        """
        if self.is_gaussian_prediction:
            crps = gaussian_crps(target_data.response, predictions.mean, predictions.std)
            rmse = np.sqrt(np.mean((predictions.mean - target_data.response) ** 2))
        else:
            target_npbd = ModelDataAdapter.to_npboost_data(target_data)
            mse, rmse = point_pred_eval(predictions, target_npbd)
            crps = crps_eval(predictions, target_npbd, self.crps_samples, self.device)
        return {"crps": crps, "rmse": rmse}

    def _evaluate(self, predictions: GroupedModelPrediction, target_npbd: NPBDataset) -> Dict[str, float]:
        """Evaluate grouped NP-style predictions against an already converted target."""
        mse, rmse = point_pred_eval(predictions, target_npbd)
        crps = crps_eval(predictions, target_npbd, self.crps_samples, self.device)
        return {"crps": crps, "rmse": rmse}

    @abstractmethod
    def save_model(self, model_folder: str, model_file_name: str):
        """Serialize the fitted runner/model state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "ExperimentRunner":
        """Deserialize a runner/model state created by ``save_model``."""
        pass

    def _wandb_plot_key(self, plot_name: str, plot_kind: str, step_value: int) -> str:
        """Build stable W&B media keys grouped by model, plot type, and step."""
        return f"plots/{plot_name}/{plot_kind}/{step_value:03d}"

    def _wandb_image(self, plot, caption: str):
        """Render a plot object to a W&B image."""
        figure = plot.draw()
        try:
            return wandb.Image(figure, caption=caption)
        finally:
            plt.close(figure)

    def _log_wandb_media(self, media: Dict[str, Any], step: Optional[int] = None):
        """Log collected media only when W&B is enabled."""
        if self.wandb_enabled and media:
            wandb.log(media, step=step)


class NPExperiment(ExperimentRunner):
    """Runner for the NP and ANP experiments.

    See ``np_trainer.py`` file for more details on the NP training.  
    See ``neural_process.py`` file for the NP architecture.
    """

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, device: str, 
                 crps_samples: int, wandb_enabled: bool, max_epochs: int, validation_metric: str,
                 min_delta: float, patience: int, restore_best_model: bool,
                 params: DictConfig, train_params: DictConfig, plot_intermediate: bool = False):
        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)
        self.device = torch.device(device)
        self.crps_samples = crps_samples
        self.wandb_enabled = wandb_enabled

        self.max_epochs = max_epochs
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_model = restore_best_model

        self.params = params
        self.train_params = train_params
        self.loss_fn = NPMLLoss()
        self.is_gaussian_prediction = False

    def fit(self, train_data: BaseDataset, validation_target_data: BaseDataset, validation_context_data: Optional[BaseDataset] = None):
        train_npbd = ModelDataAdapter.to_np_data(train_data)
        validation_context_npbd = ModelDataAdapter.to_np_data(validation_context_data) if validation_context_data is not None else None
        validation_target_npbd = ModelDataAdapter.to_np_data(validation_target_data) if validation_target_data is not None else None
        return self._fit(train_npbd=train_npbd, 
                         validation_target_npbd=validation_target_npbd, 
                         validation_context_npbd=validation_context_npbd)

    def _fit(self, train_npbd: NPBDataset, validation_target_npbd: Optional[NPBDataset] = None, validation_context_npbd: Optional[NPBDataset] = None):
        self.logger.info(f"Starting NP experiment for task {self.task.name}")
        
        self.np_model = NeuralProcess(x_dim=train_npbd.features.shape[1], **self.params)
        self.optimizer = Adam(params=self.np_model.parameters(), lr=self.train_params.learning_rate)

        self.np_trainer = NPTrainer(
            model=self.np_model,
            optimizer=self.optimizer,
            loss=self.loss_fn,
            scheduler=None,
            logger=self.logger,
            device=self.device,
            task=self.task,
            plot_intermediate=self.plot_intermediate,
        )

        self.np_trainer.train(
            train_data=train_npbd,
            validation_metric=self.validation_metric,
            validation_context=validation_context_npbd,
            validation_target=validation_target_npbd,
            max_epochs=self.max_epochs,
            batch_size=self.train_params.batch_size,
            patience=self.patience,
            min_delta=self.min_delta,
            restore_best_model=self.restore_best_model,
            crps_samples=self.crps_samples,
            wandb_enabled=self.wandb_enabled,
        )
        self.logger.info("NP experiment complete.")
        self.is_fitted = True
        return self

    def predict(self, target_data: BaseDataset, context_data: Optional[BaseDataset] = None) -> ModelPrediction:
        context_npbd = ModelDataAdapter.to_np_data(context_data)
        target_npbd = ModelDataAdapter.to_np_data(target_data)
        return self._predict(target_npbd=target_npbd, context_npbd=context_npbd)
       
    def _predict(self, target_npbd: NPBDataset, context_npbd: NPBDataset) -> ModelPrediction:
        self.logger.info("Generating predictions with Neural Process model.")
        predictions = self.np_trainer.predict(target_npbd=target_npbd, context_npbd=context_npbd)
        return predictions

    def run(self, data_bundle: DataBundle) -> Dict[str, Dict[str, float]]:
        results = {}
        train_npbd = ModelDataAdapter.to_np_data(data_bundle.train)
        if self.task.name == IN_CONTEXT_TASK_NAME:
            validation_target = data_bundle.validation
            validation_context = data_bundle.train
            test_target = data_bundle.test
            test_context = data_bundle.train
            validation_context_npbd = train_npbd
            validation_target_npbd = ModelDataAdapter.to_np_data(data_bundle.validation)
            test_context_npbd = train_npbd
            test_target_npbd = ModelDataAdapter.to_np_data(data_bundle.test)
        elif self.task.name == FEW_SHOT_TASK_NAME:
            validation_target = data_bundle.validation_query
            validation_context = data_bundle.validation_support
            test_target = data_bundle.test_query
            test_context = data_bundle.test_support
            validation_context_npbd = ModelDataAdapter.to_np_data(data_bundle.validation_support)
            validation_target_npbd = ModelDataAdapter.to_np_data(data_bundle.validation_query)
            test_context_npbd = ModelDataAdapter.to_np_data(data_bundle.test_support)
            test_target_npbd = ModelDataAdapter.to_np_data(data_bundle.test_query)
        else:
            raise ValueError(f"Unknown task name: {self.task.name}")

        start_time = time.time()
        self._fit(train_npbd=train_npbd, 
                  validation_target_npbd=validation_target_npbd, 
                  validation_context_npbd=validation_context_npbd)
        fitting_time = time.time() - start_time
        self.logger.info(f"Fitting time: {fitting_time:.2f} seconds")

        self._set_prediction_seed(VALIDATION_SPLIT_NAME)
        start_time = time.time()
        validation_predictions = self._predict(target_npbd=validation_target_npbd, context_npbd=validation_context_npbd)
        validation_prediction_time = time.time() - start_time
        self.logger.info(f"Validation prediction time: {validation_prediction_time:.2f} seconds")

        self._set_prediction_seed(TEST_SPLIT_NAME)
        start_time = time.time()
        test_predictions = self._predict(target_npbd=test_target_npbd, context_npbd=test_context_npbd)
        test_prediction_time = time.time() - start_time
        self.logger.info(f"Test prediction time: {test_prediction_time:.2f} seconds")

        if self.wandb_enabled:
            wandb.summary.update({
                "fitting_time_s": fitting_time,
                "validation_prediction_time_s": validation_prediction_time,
                "test_prediction_time_s": test_prediction_time,
            })

        self._set_evaluation_seed(VALIDATION_SPLIT_NAME)
        validation_results = self._evaluate(predictions=validation_predictions, target_npbd=validation_target_npbd)
        self._set_evaluation_seed(TEST_SPLIT_NAME)
        test_results = self._evaluate(predictions=test_predictions, target_npbd=test_target_npbd)
        
        results[VALIDATION_SPLIT_NAME] = validation_results
        results[TEST_SPLIT_NAME] = test_results

        wandb_plots = {}
        best_epoch = self.np_trainer.val_loss_best_epochs

        # Coverage plot
        coverage_ggplot = plot_single_model_coverage(
            model_name=self.name,
            predictions=validation_predictions,
            prediction_metrics=validation_results,
            target_data=validation_target_npbd,
            save_path=None,
            dpi=200,
            is_gaussian_prediction=self.is_gaussian_prediction,
        )
        
        wandb_plots[self._wandb_plot_key(self.name, "coverage/final", best_epoch)] = self._wandb_image(
                coverage_ggplot,
                caption=f"Final coverage calibration",
            )
        
        # Residual vs. predicted plot
        residual_ggplot = plot_residuals_vs_predicted_single(
            model_name=self.name,
            predictions=validation_predictions,
            prediction_metrics=validation_results,
            target_data=validation_target_npbd,
            save_path=None,
            dpi=200,
        )
        wandb_plots[self._wandb_plot_key(self.name, "residuals/final", best_epoch)] = self._wandb_image(
            residual_ggplot,
            caption=f"Final residuals vs. predicted",
        )

        if validation_target_npbd.features.shape[1] == 1:
            true_fixed_effect, true_random_effect = _combined_true_effect_lines(
                validation_context,
                validation_target,
            )
            pred_ggplot = plot_single_model_predictions(
                model_name=self.name,
                predictions=validation_predictions,
                prediction_metrics=validation_results,
                target_data=validation_target_npbd,
                context_data=validation_context_npbd,
                task=self.task,
                n_groups=8,
                dpi=200,
                eval_samples=self.crps_samples,
                is_gaussian_prediction=self.is_gaussian_prediction,
                true_fixed_effect=true_fixed_effect,
                true_random_effect=true_random_effect,
            )

            wandb_plots[self._wandb_plot_key(self.name, "predictions/final", best_epoch)] = self._wandb_image(
                pred_ggplot,
                caption=f"Final predictions and uncertainty",
            )

        if test_target_npbd.features.shape[1] == 1:
            true_fixed_effect, true_random_effect = _combined_true_effect_lines(
                test_context,
                test_target,
            )
            test_pred_ggplot = plot_single_model_predictions(
                model_name=self.name,
                predictions=test_predictions,
                prediction_metrics=test_results,
                target_data=test_target_npbd,
                context_data=test_context_npbd,
                task=self.task,
                n_groups=8,
                dpi=200,
                eval_samples=self.crps_samples,
                is_gaussian_prediction=self.is_gaussian_prediction,
                true_fixed_effect=true_fixed_effect,
                true_random_effect=true_random_effect,
                prediction_split_name="Test",
            )

            wandb_plots[self._wandb_plot_key(self.name, "predictions/test/final", best_epoch)] = self._wandb_image(
                test_pred_ggplot,
                caption="Final test predictions and uncertainty",
            )

        if _supports_fixed_effect_diagnostics(self.name, validation_predictions):
            variance_ggplot = plot_variance_components(self.name, validation_predictions, dpi=200)
            wandb_plots[self._wandb_plot_key(self.name, "variance_components/final", best_epoch)] = self._wandb_image(
                variance_ggplot,
                caption="Final variance decomposition",
            )

            density_ggplot = plot_fixed_effect_density(self.name, validation_predictions, dpi=200)
            wandb_plots[self._wandb_plot_key(self.name, "fixed_effect_density/final", best_epoch)] = self._wandb_image(
                density_ggplot,
                caption="Final fixed effect density",
            )

        self._log_wandb_media(wandb_plots, step=self.np_trainer.epochs_trained)
        if self.wandb_enabled:
            wandb.summary.update(self.np_trainer._sigma_e_summary())
            
        return results

    def save_model(self, model_folder: str, model_file_name: str):
        model_path = Path(f"{model_folder}/{model_file_name}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "NPExperiment":
        if not torch.cuda.is_available():
            try:
                with force_torch_cpu_load():
                    instance = joblib.load(f"{model_folder}/{model_file_name}.pkl")

                if torch.device(instance.device) == torch.device("cuda"):
                    instance.logger.warning(
                        "Model was trained on GPU but no GPU is available. Loading on CPU instead."
                    )
                    instance.device = torch.device("cpu")
                    instance.np_trainer.device = instance.device
            except Exception as e:
                raise e
        else:
            instance = joblib.load(f"{model_folder}/{model_file_name}.pkl")
        return instance


class NPBoostExperiment(ExperimentRunner):
    """Runner for NPBoost experiments.
    
    See ``npboost.py`` file for details on the NPBoost algorithm and training.
    See ``neural_process.py`` file for the NP architecture.
    """

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, validation_metric: str, wandb_enabled: bool, device: str,
                 crps_samples: int, max_boosting_rounds: int, min_delta: float,
                 patience: int, restore_best_model: bool, np_params: DictConfig, lgbm_params: DictConfig, np_train_params: DictConfig,
                 fixed_effect_warm_start_rounds: int = 0, compute_aux_validation_metrics: bool = True, plot_intermediate: bool = False,
                 npboost_model: Optional[NPBoost] = None, logger_name: str = "NPBoostExperiment"):
        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)
        self.npboost = npboost_model or NPBoost(
            name=name,
            experiment_seed=experiment_seed,
            task=task,
            validation_metric=validation_metric,
            device=device,
            crps_samples=crps_samples,
            max_boosting_rounds=max_boosting_rounds,
            min_delta=min_delta,
            patience=patience,
            restore_best_model=restore_best_model,
            np_params=np_params,
            lgbm_params=lgbm_params,
            np_train_params=np_train_params,
            fixed_effect_warm_start_rounds=fixed_effect_warm_start_rounds,
            compute_aux_validation_metrics=compute_aux_validation_metrics,
            plot_intermediate=plot_intermediate,
        )
        self.device = self.npboost.device
        self.crps_samples = self.npboost.crps_samples
        self.wandb_enabled = wandb_enabled
        self.is_gaussian_prediction = False
        self.logger = logging.getLogger(logger_name)

    def __getattr__(self, name: str):
        """Forward unknown attributes to ``NPBoost`` instance.
        
        This makes it possible to access the underlying NPBoost attributes directly from the runner."""
        npboost = self.__dict__.get("npboost")
        if npboost is not None:
            try:
                return getattr(npboost, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        """Keep mirrored attributes on the wrapper and wrapped model in sync."""
        npboost = self.__dict__.get("npboost")
        if name != "npboost" and npboost is not None and hasattr(npboost, name):
            setattr(npboost, name, value)
        object.__setattr__(self, name, value)

    def run(self, data_bundle: DataBundle) -> Dict[str, Dict[str, float]]:
        results = {}
        train_npbd = ModelDataAdapter.to_npboost_data(data_bundle.train)
        if self.task.name == IN_CONTEXT_TASK_NAME:
            validation_context = data_bundle.train
            validation_target = data_bundle.validation
            test_context = data_bundle.train
            test_target = data_bundle.test
            validation_context_npbd = train_npbd
            validation_target_npbd = ModelDataAdapter.to_npboost_data(data_bundle.validation)
            test_context_npbd = train_npbd
            test_target_npbd = ModelDataAdapter.to_npboost_data(data_bundle.test)
        elif self.task.name == FEW_SHOT_TASK_NAME:
            validation_context = data_bundle.validation_support
            validation_target = data_bundle.validation_query
            test_context = data_bundle.test_support
            test_target = data_bundle.test_query
            validation_context_npbd = ModelDataAdapter.to_npboost_data(data_bundle.validation_support)
            validation_target_npbd = ModelDataAdapter.to_npboost_data(data_bundle.validation_query)
            test_context_npbd = ModelDataAdapter.to_npboost_data(data_bundle.test_support)
            test_target_npbd = ModelDataAdapter.to_npboost_data(data_bundle.test_query)
        else:
            raise ValueError(f"Unknown task name: {self.task.name}")

        start_time = time.time()
        self._fit(
            train_npbd=train_npbd,
            validation_context_npbd=validation_context_npbd,
            validation_target_npbd=validation_target_npbd,
        )
        fitting_time = time.time() - start_time
        self.logger.info(f"Fitting time: {fitting_time:.2f} seconds")

        self._set_prediction_seed(VALIDATION_SPLIT_NAME)
        start_time = time.time()
        validation_predictions = self._predict(target_npbd=validation_target_npbd, context_npbd=validation_context_npbd)
        validation_prediction_time = time.time() - start_time
        self.logger.info(f"Validation prediction time: {validation_prediction_time:.2f} seconds")

        self._set_prediction_seed(TEST_SPLIT_NAME)
        start_time = time.time()
        test_predictions = self._predict(target_npbd=test_target_npbd, context_npbd=test_context_npbd)
        test_prediction_time = time.time() - start_time
        self.logger.info(f"Test prediction time: {test_prediction_time:.2f} seconds")

        if self.wandb_enabled:
            wandb.summary.update({
                "fitting_time_s": fitting_time,
                "validation_prediction_time_s": validation_prediction_time,
                "test_prediction_time_s": test_prediction_time,
            })

        self.logger.info("Evaluating model on validation data.")
        self._set_evaluation_seed(VALIDATION_SPLIT_NAME)
        validation_results = self._evaluate(validation_predictions, validation_target_npbd)
        validation_results["fixed_effect_residual_rmse"] = self._fixed_effect_residual_rmse_from_prediction_arrays(
            validation_predictions,
            validation_target_npbd,
        )
        self.logger.info("Evaluating model on test data.")
        self._set_evaluation_seed(TEST_SPLIT_NAME)
        test_results = self._evaluate(test_predictions, test_target_npbd)
        test_results["fixed_effect_residual_rmse"] = self._fixed_effect_residual_rmse_from_prediction_arrays(
            test_predictions,
            test_target_npbd,
        )
        self._log_final_wandb_plots(
            validation_predictions=validation_predictions,
            validation_results=validation_results,
            validation_target=validation_target,
            validation_target_npbd=validation_target_npbd,
            validation_context=validation_context,
            validation_context_npbd=validation_context_npbd,
            test_predictions=test_predictions,
            test_results=test_results,
            test_target_npbd=test_target_npbd,
            test_target=test_target,
            test_context=test_context,
            test_context_npbd=test_context_npbd,
        )

        results[VALIDATION_SPLIT_NAME] = validation_results
        results[TEST_SPLIT_NAME] = test_results
        return results

    def fit(self, train_data: BaseDataset, validation_target_data: BaseDataset, validation_context_data: BaseDataset) -> "NPBoostExperiment":
        self.logger.info(f"Starting {self.name} training...")
        train_npbd = ModelDataAdapter.to_npboost_data(train_data)
        validation_target_npbd = ModelDataAdapter.to_npboost_data(validation_target_data)
        validation_context_npbd = (
            ModelDataAdapter.to_npboost_data(validation_context_data)
            if validation_context_data is not None
            else None
        )
        return self._fit(
            train_npbd=train_npbd,
            validation_target_npbd=validation_target_npbd,
            validation_context_npbd=validation_context_npbd,
        )

    def _fit(self, train_npbd: NPBDataset, validation_target_npbd: NPBDataset, validation_context_npbd: NPBDataset) -> "NPBoostExperiment":
        self.npboost.fit(
            train_npbd=train_npbd,
            validation_target_npbd=validation_target_npbd,
            validation_context_npbd=validation_context_npbd,
            log_callback=self._log_training_metrics,
            validation_callback=self._log_intermediate_validation_plots,
        )
        self.is_fitted = self.npboost.is_fitted
        self.device = self.npboost.device
        if self.wandb_enabled:
            wandb.summary.update({
                f"val_{self.validation_metric}_best": self.npboost.val_best,
                "best_boosting_round": self.npboost.val_metric_best_boosting_round,
                "early_stopped": self.npboost.early_stopped,
                "restored_best_model": self.npboost.restore_best_model,
                "n_boosting_rounds_trained": self.npboost.n_boosting_rounds_trained,
            })
            wandb.summary.update(self.npboost.sigma_e_summary())
        return self

    def predict(self, target_data: BaseDataset, context_data: Optional[BaseDataset] = None) -> GroupedModelPrediction:
        context_npbd = ModelDataAdapter.to_npboost_data(context_data)
        target_npbd = ModelDataAdapter.to_npboost_data(target_data)
        return self._predict(target_npbd=target_npbd, context_npbd=context_npbd)

    def _predict(self, target_npbd: NPBDataset, context_npbd: NPBDataset) -> GroupedModelPrediction:
        return self.npboost.predict(target_npbd=target_npbd, context_npbd=context_npbd)

    def _evaluate(self, predictions: GroupedModelPrediction, target_npbd: NPBDataset) -> Dict[str, float]:
        mse, rmse = point_pred_eval(predictions, target_npbd)
        crps = crps_eval(predictions, target_npbd, self.crps_samples, self.device)
        return {"crps": crps, "rmse": rmse}

    @staticmethod
    def _rmse_from_prediction_arrays(predictions: GroupedModelPrediction, target_npbd: NPBDataset) -> float:
        return NPBoost._rmse_from_prediction_arrays(predictions, target_npbd)

    @staticmethod
    def _fixed_effect_residual_rmse_from_prediction_arrays(predictions: GroupedModelPrediction, target_npbd: NPBDataset) -> float:
        return NPBoost._fixed_effect_residual_rmse_from_prediction_arrays(predictions, target_npbd)

    def _log_training_metrics(self, step: int, metrics: Dict[str, float]):
        if self.wandb_enabled:
            wandb.log(metrics, step=step)

    def _log_intermediate_validation_plots(
        self,
        model: NPBoost,
        validation_context: NPBDataset,
        validation_target: NPBDataset,
        validation_predictions: GroupedModelPrediction,
        validation_results: Dict[str, float],
    ):
        """Log selected validation diagnostics during boosting."""
        if model.current_boosting_round not in INTERMEDIATE_PLOTTING_ROUNDS or not self.plot_intermediate:
            return

        wandb_plots = {}
        coverage_ggplot = plot_single_model_coverage(
            model_name=self.name,
            predictions=validation_predictions,
            prediction_metrics=validation_results,
            target_data=validation_target,
            save_path=None,
            dpi=200,
            show=False,
            eval_samples=self.crps_samples,
            is_gaussian_prediction=self.is_gaussian_prediction,
        )
        wandb_plots[self._wandb_plot_key(self.name, "coverage", model.current_boosting_round)] = self._wandb_image(
            coverage_ggplot,
            caption=f"Coverage calibration for boosting round {model.current_boosting_round}",
        )

        if _supports_fixed_effect_diagnostics(self.name, validation_predictions):
            fe_vs_resp_ggplot = plot_fixed_effect_vs_response(
                val_preds=validation_predictions,
                val_target=validation_target,
                test_preds=validation_predictions,
                test_target=validation_target,
                model_name=self.name,
                dpi=200,
            )
            wandb_plots[self._wandb_plot_key(self.name, "fixed_effect_vs_response", model.current_boosting_round)] = self._wandb_image(
                fe_vs_resp_ggplot,
                caption=f"Predicted fixed effect vs. true response for boosting round {model.current_boosting_round}",
            )

            variance_ggplot = plot_variance_components(self.name, validation_predictions, dpi=200)
            wandb_plots[self._wandb_plot_key(self.name, "variance_components", model.current_boosting_round)] = self._wandb_image(
                variance_ggplot,
                caption=f"Variance decomposition for boosting round {model.current_boosting_round}",
            )

        if _supports_fixed_effect_diagnostics(self.name, validation_predictions) and validation_target.features.shape[1] <= 2:
            fixed_effect_ggplot = plot_fixed_effect(
                features=validation_target.features,
                fixed_effect_pred=validation_predictions.fixed_effect,
                model_name=self.name,
                dpi=200,
            )
            wandb_plots[self._wandb_plot_key(self.name, "fixed_effect", model.current_boosting_round)] = self._wandb_image(
                fixed_effect_ggplot,
                caption=f"Fixed effect estimate for boosting round {model.current_boosting_round}",
            )

        if validation_target.features.shape[1] == 1:
            pred_ggplot = plot_single_model_predictions(
                model_name=self.name,
                predictions=validation_predictions,
                prediction_metrics=validation_results,
                target_data=validation_target,
                context_data=validation_context,
                task=self.task,
                n_groups=8,
                dpi=200,
                eval_samples=self.crps_samples,
                is_gaussian_prediction=self.is_gaussian_prediction,
            )
            wandb_plots[self._wandb_plot_key(self.name, "predictions", model.current_boosting_round)] = self._wandb_image(
                pred_ggplot,
                caption=f"Predictions and uncertainty for boosting round {model.current_boosting_round}",
            )

        self._log_wandb_media(wandb_plots, step=model.current_boosting_round)

    def _log_final_wandb_plots(
        self,
        validation_predictions: GroupedModelPrediction,
        validation_results: Dict[str, float],
        validation_target: BaseDataset,
        validation_target_npbd: NPBDataset,
        validation_context: BaseDataset,
        validation_context_npbd: NPBDataset,
        test_predictions: GroupedModelPrediction,
        test_results: Dict[str, float],
        test_target_npbd: NPBDataset,
        test_target: BaseDataset,
        test_context: BaseDataset,
        test_context_npbd: NPBDataset,
    ):
        """Log the final validation/test diagnostic plots after NPBoost training."""
        wandb_plots = {}
        step_value = self.npboost.val_metric_best_boosting_round

        coverage_ggplot = plot_single_model_coverage(
            model_name=self.name,
            predictions=validation_predictions,
            prediction_metrics=validation_results,
            target_data=validation_target,
            save_path=None,
            dpi=200,
            is_gaussian_prediction=self.is_gaussian_prediction,
        )
        wandb_plots[self._wandb_plot_key(self.name, "coverage/final", step_value)] = self._wandb_image(
            coverage_ggplot,
            caption="Final coverage calibration",
        )

        residual_ggplot = plot_residuals_vs_predicted_single(
            model_name=self.name,
            predictions=validation_predictions,
            prediction_metrics=validation_results,
            target_data=validation_target,
            save_path=None,
            dpi=200,
        )
        wandb_plots[self._wandb_plot_key(self.name, "residuals/final", step_value)] = self._wandb_image(
            residual_ggplot,
            caption="Final residuals vs. predicted",
        )

        if validation_target_npbd.features.shape[1] <= 2:
            fixed_effect_ggplot = plot_fixed_effect(
                features=validation_target_npbd.features,
                fixed_effect_pred=validation_predictions.fixed_effect,
                true_fixed_effect=validation_target.fixed_effect_part,
                model_name=self.name,
            )
            wandb_plots[self._wandb_plot_key(self.name, "fixed_effect/final", step_value)] = self._wandb_image(
                fixed_effect_ggplot,
                caption="Final estimated fixed effect",
            )

        if validation_target_npbd.features.shape[1] == 1:
            true_fixed_effect, true_random_effect = _combined_true_effect_lines(
                validation_context,
                validation_target,
            )

            pred_ggplot = plot_single_model_predictions(
                model_name=self.name,
                predictions=validation_predictions,
                prediction_metrics=validation_results,
                target_data=validation_target_npbd,
                context_data=validation_context_npbd,
                task=self.task,
                n_groups=8,
                dpi=200,
                eval_samples=self.crps_samples,
                is_gaussian_prediction=self.is_gaussian_prediction,
                true_fixed_effect=true_fixed_effect,
                true_random_effect=true_random_effect,
            )
            wandb_plots[self._wandb_plot_key(self.name, "predictions/final", step_value)] = self._wandb_image(
                pred_ggplot,
                caption="Final predictions and uncertainty",
            )

        if test_target_npbd.features.shape[1] == 1:
            true_fixed_effect, true_random_effect = _combined_true_effect_lines(
                test_context,
                test_target,
            )
            test_pred_ggplot = plot_single_model_predictions(
                model_name=self.name,
                predictions=test_predictions,
                prediction_metrics=test_results,
                target_data=test_target_npbd,
                context_data=test_context_npbd,
                task=self.task,
                n_groups=8,
                dpi=200,
                eval_samples=self.crps_samples,
                is_gaussian_prediction=self.is_gaussian_prediction,
                true_fixed_effect=true_fixed_effect,
                true_random_effect=true_random_effect,
                prediction_split_name="Test",
            )
            wandb_plots[self._wandb_plot_key(self.name, "predictions/test/final", step_value)] = self._wandb_image(
                test_pred_ggplot,
                caption="Final test predictions and uncertainty",
            )

        if _supports_fixed_effect_diagnostics(self.name, validation_predictions):
            variance_ggplot = plot_variance_components(self.name, validation_predictions, dpi=200)
            wandb_plots[self._wandb_plot_key(self.name, "variance_components/final", step_value)] = self._wandb_image(
                variance_ggplot,
                caption="Final variance decomposition",
            )

            density_ggplot = plot_fixed_effect_density(self.name, validation_predictions, dpi=200)
            wandb_plots[self._wandb_plot_key(self.name, "fixed_effect_density/final", step_value)] = self._wandb_image(
                density_ggplot,
                caption="Final fixed effect density",
            )

        if (
            _supports_fixed_effect_diagnostics(self.name, validation_predictions)
            and _supports_fixed_effect_diagnostics(self.name, test_predictions)
        ):
            fe_vs_resp_ggplot = plot_fixed_effect_vs_response(
                val_preds=validation_predictions,
                val_target=validation_target,
                test_preds=test_predictions,
                test_target=test_target,
                model_name=self.name,
                dpi=200,
            )
            wandb_plots[self._wandb_plot_key(self.name, "fixed_effect_vs_response/final", step_value)] = self._wandb_image(
                fe_vs_resp_ggplot,
                caption="Final predicted fixed effect vs. true response",
            )

        self._log_wandb_media(wandb_plots, step=self.npboost.n_boosting_rounds_trained)

    def save_model(self, model_folder: str, model_file_name: str):
        model_path = Path(f"{model_folder}/{model_file_name}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "NPBoostExperiment":
        if not torch.cuda.is_available():
            try:
                with force_torch_cpu_load():
                    instance = joblib.load(f"{model_folder}/{model_file_name}.pkl")

                if instance.device == torch.device("cuda"):
                    instance.logger.warning(
                        "Model was trained on GPU but no GPU is available. Loading on CPU instead."
                    )
                    instance.device = torch.device("cpu")
                    instance.npboost.device = instance.device
                    instance.npboost.np_trainer.device = instance.device
            except Exception as e:
                raise e
        else:
            instance = joblib.load(f"{model_folder}/{model_file_name}.pkl")
        return instance

class GPExperiment(ExperimentRunner):
    """Base class for all mixed-effects models experiments."""

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, validation_metric: str, plot_intermediate: bool = False):
        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)
        # These model's predictive distributions are Gaussian.
        self.is_gaussian_prediction = True


class GradientBoostingExperiment(GPExperiment):
    """Plain LightGBM gradient boosting baseline with optional group ID feature."""

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, wandb_enabled: bool = False,
                 group_as_cat: bool = False, max_boosting_rounds: int = 500, train_params: DictConfig = None,
                 validation_metric: str = "rmse", min_delta: float = 0.0, patience: int = 25,
                 restore_best_model: bool = True, plot_intermediate: bool = False):
        if validation_metric != "rmse":
            raise ValueError("GradientBoostingExperiment only supports validation_metric='rmse'.")
        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)
        self.wandb_enabled = wandb_enabled
        self.group_as_cat = group_as_cat
        self.max_boosting_rounds = max_boosting_rounds
        self.train_params = train_params
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_model = restore_best_model
        self.bst = None
        self.group_categories = None

    def _features(self, data: BaseDataset, fit: bool = False) -> pd.DataFrame:
        """Return LightGBM features, optionally adding a categorical group column."""
        features = data.features.copy()
        if not self.group_as_cat:
            return features

        if fit:
            self.group_categories = pd.Index(pd.unique(data.grouping_var))
        elif self.group_categories is None:
            raise ValueError("Model must be fitted before preparing prediction features.")

        features[GROUPING_COLUMN_NAME] = pd.Categorical(
            data.grouping_var,
            categories=self.group_categories,
        )
        return features

    def fit(self, train_data: BaseDataset, validation_target_data: BaseDataset, validation_context_data: Optional[BaseDataset] = None):
        train_features = self._features(train_data, fit=True)
        validation_features = self._features(validation_target_data)
        train_set = Dataset(
            data=train_features,
            label=train_data.response.values,
            categorical_feature=[GROUPING_COLUMN_NAME] if self.group_as_cat else [],
            free_raw_data=False,
        )

        self.early_stopped = False
        self.patience_counter = 0
        self.val_metric_best = float("inf")
        self.val_metric_best_boosting_round = 0
        self.n_boosting_rounds_trained = self.max_boosting_rounds
        best_bst_str = None
        self.bst = None

        params = dict(self.train_params) if self.train_params is not None else {}
        params.pop("line_search_step_length", None)

        callback_state = {"best_bst_str": None}

        def custom_validation_callback(env):
            self.bst = env.model
            self.n_boosting_rounds_trained = env.iteration + 1
            logger.info(f"Boosting round {self.n_boosting_rounds_trained}/{self.max_boosting_rounds}")

            validation_mean = self.bst.predict(validation_features)
            validation_rmse = float(np.sqrt(np.mean((validation_mean - validation_target_data.response.values) ** 2)))

            if self.wandb_enabled:
                wandb.log({"val_rmse": validation_rmse, "current_val_metric": validation_rmse}, step=self.n_boosting_rounds_trained)

            if validation_rmse < (self.val_metric_best - self.min_delta):
                self.val_metric_best = validation_rmse
                self.val_metric_best_boosting_round = self.n_boosting_rounds_trained
                self.patience_counter = 0
                if self.restore_best_model:
                    callback_state["best_bst_str"] = self.bst.model_to_string()
            else:
                self.patience_counter += 1

            logger.info(
                f"Patience: {self.patience_counter}/{self.patience}. "
                f"Current RMSE: {validation_rmse:.6f} | Best: {self.val_metric_best:.6f}"
            )

            if self.patience_counter >= self.patience:
                self.early_stopped = True
                logger.info(f"[Early Stopping] No improvement for {self.patience} rounds.")
                raise LightGBMEarlyStopException(
                    best_iteration=self.val_metric_best_boosting_round - 1,
                    best_score=[],
                )

        custom_validation_callback.order = 30
        custom_validation_callback.before_iteration = False

        self.bst = lgb_train(
            params=params,
            train_set=train_set,
            num_boost_round=self.max_boosting_rounds,
            callbacks=[custom_validation_callback],
        )

        if self.restore_best_model and callback_state["best_bst_str"] is not None:
            self.bst = Booster(model_str=callback_state["best_bst_str"])
            logger.info(f"Restored model from best round {self.val_metric_best_boosting_round}.")

        if self.wandb_enabled:
            wandb.summary.update({
                "val_rmse": self.val_metric_best,
                "best_boosting_round": self.val_metric_best_boosting_round,
                "early_stopped": self.early_stopped,
                "restored_best_model": self.restore_best_model,
                "n_boosting_rounds_trained": self.n_boosting_rounds_trained,
            })

        self.is_fitted = True
        return self

    def predict(self, target_data: BaseDataset, context_data: Optional[BaseDataset] = None) -> ModelPrediction:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        target_features = self._features(target_data)
        mean = self.bst.predict(target_features)
        return ModelPrediction(
            mean=mean,
            std=np.zeros_like(mean, dtype=float),
            group_ids=target_data.grouping_var.values,
            fixed_effect=mean,
            random_effect=np.zeros_like(mean, dtype=float),
        )

    def evaluate(self, predictions: ModelPrediction, target_data: BaseDataset) -> Dict[str, float]:
        rmse = float(np.sqrt(np.mean((predictions.mean - target_data.response.values) ** 2)))
        return {"rmse": rmse}

    def save_model(self, model_folder: str, model_file_name: str):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        model_path = Path(f"{model_folder}/{model_file_name}.txt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.bst.save_model(str(model_path))
        other_variables = {
            "train_params": self.train_params,
            "experiment_seed": self.experiment_seed,
            "name": self.name,
            "task": self.task,
            "group_as_cat": self.group_as_cat,
            "group_categories": self.group_categories,
            "early_stopped": self.early_stopped,
            "patience_counter": self.patience_counter,
            "val_metric_best": self.val_metric_best,
            "val_metric_best_boosting_round": self.val_metric_best_boosting_round,
            "n_boosting_rounds_trained": self.n_boosting_rounds_trained,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best_model": self.restore_best_model,
            "max_boosting_rounds": self.max_boosting_rounds,
            "wandb_enabled": self.wandb_enabled,
        }
        joblib.dump(other_variables, f"{model_folder}/{model_file_name}_vars.pkl")

    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "GradientBoostingExperiment":
        instance = cls.__new__(cls)
        instance.bst = Booster(model_file=f"{model_folder}/{model_file_name}.txt")
        vars_path = Path(f"{model_folder}/{model_file_name}_vars.pkl")
        other_variables = joblib.load(vars_path)
        for key, value in other_variables.items():
            setattr(instance, key, value)
        instance.validation_metric = "rmse"
        instance.is_fitted = True
        instance.is_gaussian_prediction = True
        instance.device = torch.device("cpu")
        instance.crps_samples = 20
        instance.plot_intermediate = False
        instance.logger = logging.getLogger("GradientBoostingExperiment")
        return instance


class LMEExperiment(GPExperiment):
    """Linear mixed-effects runner.

    In-context predictions use the fitted training random-effect model directly.
    Few-shot predictions condition a temporary random-effect model on support
    residuals, while keeping the fitted fixed-effect coefficients and covariance.
    """

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, validation_metric: str = "rmse", plot_intermediate: bool = False):
        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)

    def run(self, data_bundle: DataBundle) -> Dict[str, Dict[str, float]]:
        results = {}
        train_data = data_bundle.train
        if self.task.name == IN_CONTEXT_TASK_NAME:
            validation_context_data = None
            validation_target_data = data_bundle.validation
            test_context_data = None
            test_target_data = data_bundle.test
        elif self.task.name == FEW_SHOT_TASK_NAME:
            validation_context_data = data_bundle.validation_support
            validation_target_data = data_bundle.validation_query
            test_context_data = data_bundle.test_support
            test_target_data = data_bundle.test_query
        else:
            raise ValueError(f"Unknown task name: {self.task.name}")

        self.fit(train_data=train_data)
        self._set_prediction_seed(VALIDATION_SPLIT_NAME)
        validation_predictions = self.predict(target_data=validation_target_data, context_data=validation_context_data)
        self._set_prediction_seed(TEST_SPLIT_NAME)
        test_predictions = self.predict(target_data=test_target_data, context_data=test_context_data)

        self._set_evaluation_seed(VALIDATION_SPLIT_NAME)
        validation_results = self.evaluate(validation_predictions, validation_target_data)
        self._set_evaluation_seed(TEST_SPLIT_NAME)
        test_results = self.evaluate(test_predictions, test_target_data)

        results[VALIDATION_SPLIT_NAME] = validation_results
        results[TEST_SPLIT_NAME] = test_results

        return results

    def fit(self, train_data: BaseDataset):
        (
            train_fixed_effect_features,
            train_random_effect_features,
            train_group_data,
            drop_intercept,
            train_response,
        ) = ModelDataAdapter.to_lme_data(train_data)
        
        self.train_gp_model = GPModel(
            likelihood="gaussian",
            group_data=train_group_data,
            group_rand_coef_data=train_random_effect_features,
            ind_effect_group_rand_coef=[1] * train_random_effect_features.shape[1],
            drop_intercept_group_rand_effect=drop_intercept,
            seed=self.experiment_seed,
        )
        logger.info("Starting LME fitting...")
        self.train_gp_model.fit(y=train_response, X=train_fixed_effect_features)
        logger.info("Fitted LME")
        self.best_cov_params = self.train_gp_model.get_cov_pars().values.ravel()
        self.linear_coefs = self.train_gp_model.get_coef().values.ravel()
        logger.info("LME fitting complete.")
        self.is_fitted = True
    
    def predict(self, target_data: BaseDataset, context_data: Optional[BaseDataset] = None) -> ModelPrediction:
        (
            target_fixed_effect_features,
            target_random_effect_features,
            target_group_data,
            target_drop_intercept,
            target_response,
        ) = ModelDataAdapter.to_lme_data(target_data)

        if not context_data:
            fixed_effect_target = target_fixed_effect_features @ self.linear_coefs

            target_pred = self.train_gp_model.predict(
                predict_response=True,
                predict_var=True,
                group_data_pred=target_group_data,
                group_rand_coef_data_pred=target_random_effect_features,
                X_pred=target_fixed_effect_features,
            )

            pred_mean = target_pred["mu"]
            pred_sigma = np.sqrt(target_pred["var"])
            random_effect_target = pred_mean - fixed_effect_target
        else:
            (
                context_fixed_effect_features,
                context_random_effect_features,
                context_group_data,
                context_drop_intercept,
                context_response,
            ) = ModelDataAdapter.to_lme_data(context_data)

            # 2. Calculate the fixed effect (X * beta) for the context points.
            fixed_effect_pred = context_fixed_effect_features @ self.linear_coefs

            # 3. Calculate residuals for context points to condition the random effect.
            residuals_context = context_response - fixed_effect_pred

            # 4. Calculate fixed effects for the entire target test set.
            fixed_effect_target = target_fixed_effect_features @ self.linear_coefs

            gp_model_context = GPModel(
                likelihood="gaussian",
                group_data=context_group_data,
                group_rand_coef_data=context_random_effect_features,
                ind_effect_group_rand_coef=[1] * context_random_effect_features.shape[1],
                drop_intercept_group_rand_effect=context_drop_intercept,
                seed=self.experiment_seed
            )
            random_effect_pred = gp_model_context.predict(
                predict_response=True,
                predict_var=True,
                y=residuals_context,
                cov_pars=self.best_cov_params,
                group_data_pred=target_group_data,
                group_rand_coef_data_pred=target_random_effect_features
            )

            pred_mean = fixed_effect_target + random_effect_pred["mu"]
            pred_sigma = np.sqrt(random_effect_pred["var"])
            random_effect_target = random_effect_pred["mu"]

        return ModelPrediction(
            mean=pred_mean, 
            std=pred_sigma, 
            group_ids=target_group_data,
            fixed_effect=fixed_effect_target,
            random_effect=random_effect_target
        )

    def save_model(self, model_folder: str, model_file_name: str):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        model_path = Path(model_folder) / f"{model_file_name}_gp_model.json"
        vars_path = Path(model_folder) / f"{model_file_name}_vars.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self.train_gp_model.save_model(str(model_path))
        state = {
            "experiment_seed": self.experiment_seed,
            "name": self.name,
            "task": self.task,
            "validation_metric": self.validation_metric,
            "best_cov_params": self.best_cov_params,
            "linear_coefs": self.linear_coefs,
        }
        joblib.dump(state, vars_path)
    
    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "LMEExperiment":
        instance = cls.__new__(cls)
        gp_model_path = Path(model_folder) / f"{model_file_name}_gp_model.json"
        vars_path = Path(model_folder) / f"{model_file_name}_vars.pkl"

        state = joblib.load(vars_path)
        instance.train_gp_model = GPModel(model_file=str(gp_model_path))
        
        instance.experiment_seed = state["experiment_seed"]
        instance.name = state["name"]
        instance.task = state["task"]
        instance.validation_metric = state["validation_metric"]
        instance.best_cov_params = state["best_cov_params"]
        instance.linear_coefs = state["linear_coefs"]
        
        instance.is_fitted = True
        instance.is_gaussian_prediction = True
        instance.device = torch.device("cpu")
        instance.crps_samples = 20
        instance.plot_intermediate = False
        instance.logger = logging.getLogger("LMEExperiment")
        return instance


class GPLinearExperiment(GPExperiment):
    """Runner for the GPLinear experiments.
    """

    def __init__(self, name: str, experiment_seed: int, task: BaseTask, kernel: str, validation_metric: str="rmse", plot_intermediate: bool=False):
        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)
        self.kernel = kernel

    def run(self, data_bundle: DataBundle) -> Dict[str, Dict[str, float]]:
        results = {}
        train_data = data_bundle.train
        if self.task.name == IN_CONTEXT_TASK_NAME:
            validation_context_data = None
            validation_target_data = data_bundle.validation
            test_context_data = None
            test_target_data = data_bundle.test
        elif self.task.name == FEW_SHOT_TASK_NAME:
            validation_context_data = data_bundle.validation_support
            validation_target_data = data_bundle.validation_query
            test_context_data = data_bundle.test_support
            test_target_data = data_bundle.test_query
        else:
            raise ValueError(f"Unknown task name: {self.task.name}")
        
        self.fit(train_data=train_data)
        self._set_prediction_seed(VALIDATION_SPLIT_NAME)
        validation_predictions = self.predict(target_data=validation_target_data, context_data=validation_context_data)
        self._set_prediction_seed(TEST_SPLIT_NAME)
        test_predictions = self.predict(target_data=test_target_data, context_data=test_context_data)

        self._set_evaluation_seed(VALIDATION_SPLIT_NAME)
        validation_results = self.evaluate(validation_predictions, validation_target_data)
        self._set_evaluation_seed(TEST_SPLIT_NAME)
        test_results = self.evaluate(test_predictions, test_target_data)

        results[VALIDATION_SPLIT_NAME] = validation_results
        results[TEST_SPLIT_NAME] = test_results

        return results

    def fit(self, train_data: BaseDataset):
        train_fixed_effect_features, train_gp_coords, train_group_data, train_response = ModelDataAdapter.to_gplinear_data(train_data)
        self.train_gp_model = GPModel(
            likelihood="gaussian",
            gp_coords=train_gp_coords,
            cov_function=self.kernel,
            seed=self.experiment_seed,
            cluster_ids=train_group_data,
        )
        logger.info("Starting GPLinear fitting...")
        self.train_gp_model.fit(train_response, train_fixed_effect_features)
        logger.info("Fitted GPLinear")
        self.best_cov_params = self.train_gp_model.get_cov_pars().values.ravel()
        self.linear_coefs = self.train_gp_model.get_coef().values.ravel()
        logger.info("GPLinear fitting complete.")
        self.is_fitted = True

    def predict(self, target_data: BaseDataset, context_data: Optional[BaseDataset] = None) -> ModelPrediction:
        target_fixed_effect_features, target_gp_coords, target_group_data, target_response = ModelDataAdapter.to_gplinear_data(target_data)
        if not context_data:
            fixed_effect_target = target_fixed_effect_features @ self.linear_coefs

            random_effect_pred = self.train_gp_model.predict(
                predict_response=True,
                predict_var=True,
                gp_coords_pred=target_gp_coords,
                cluster_ids_pred=target_group_data,
                X_pred=target_fixed_effect_features,
            )

            pred_mean = random_effect_pred["mu"]
            pred_sigma = np.sqrt(random_effect_pred["var"])
            random_effect_target = pred_mean - fixed_effect_target
        else:
            # 1. Get support points to serve as context for new groups.
            context_fixed_effect_features, context_gp_coords, context_group_data, context_response = ModelDataAdapter.to_gplinear_data(context_data)

            # 2. Calculate the fixed effect (X * beta) for the context points.
            fixed_effect_pred = context_fixed_effect_features @ self.linear_coefs

            # 3. Calculate residuals for context points to condition the random effect.
            residuals_context = context_response - fixed_effect_pred

            # 4. Calculate fixed effects for the entire target test set.
            fixed_effect_target = target_fixed_effect_features @ self.linear_coefs

            # 5. Predict random effects for test points, conditioned on context residuals.
            gp_model_support = GPModel(
                likelihood="gaussian",
                gp_coords=context_gp_coords,
                cov_function=self.kernel,
                seed=self.experiment_seed,
                cluster_ids=context_group_data,
            )
            random_effect_pred = gp_model_support.predict(
                predict_response=True,
                predict_var=True,
                y=residuals_context,
                cov_pars=self.best_cov_params,
                gp_coords_pred=target_gp_coords,
                cluster_ids_pred=target_group_data,
            )

            # 6. Final prediction is the sum of fixed and (conditioned) random effects.
            pred_mean = fixed_effect_target + random_effect_pred["mu"]
            pred_sigma = np.sqrt(random_effect_pred["var"])
            random_effect_target = random_effect_pred["mu"]

        return ModelPrediction(
            mean=pred_mean,
            std=pred_sigma,
            group_ids=target_group_data,
            fixed_effect=fixed_effect_target,
            random_effect=random_effect_target
        )

    def save_model(self, model_folder: str, model_file_name: str):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        model_path = Path(f"{model_folder}/{model_file_name}_gp_model.json")
        vars_path = Path(f"{model_folder}/{model_file_name}_vars.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use the GPModel's built-in save_model method to save the model parameters and state.
        self.train_gp_model.save_model(str(model_path))
        state = {
            "kernel": self.kernel,
            "experiment_seed": self.experiment_seed,
            "name": self.name,
            "task": self.task,
            "validation_metric": self.validation_metric,
            "best_cov_params": self.best_cov_params,
            "linear_coefs": self.linear_coefs,
        }
        joblib.dump(state, vars_path)
    
    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "GPLinearExperiment":
        instance = cls.__new__(cls)
        gp_model_path = Path(f"{model_folder}/{model_file_name}_gp_model.json")
        vars_path = Path(f"{model_folder}/{model_file_name}_vars.pkl")

        state = joblib.load(vars_path)
        instance.train_gp_model = GPModel(model_file=str(gp_model_path))
        
        instance.kernel = state["kernel"]
        instance.experiment_seed = state["experiment_seed"]
        instance.name = state["name"]
        instance.task = state["task"]
        instance.validation_metric = state["validation_metric"]
        instance.best_cov_params = state["best_cov_params"]
        instance.linear_coefs = state["linear_coefs"]
        
        instance.is_fitted = True
        instance.is_gaussian_prediction = True
        instance.device = torch.device("cpu")
        instance.crps_samples = 20
        instance.plot_intermediate = False
        instance.logger = logging.getLogger("GPLinearExperiment")
        return instance


@dataclass
class TabICLPrediction(ModelPrediction):
    """TabICL predictions with the quantiles needed for CRPS evaluation."""
    quantiles: Optional[np.ndarray] = None
    quantile_alphas: Optional[np.ndarray] = None


class TabICLExperiment(ExperimentRunner):
    """Runner for the TabICL_T and TabICL_P experiments

    TabICL has no training phase in the usual sense: only some context rows need to be passed.
    Then it provides predictions on the target rows.
    """

    VALID_CALL_MODES = {"taskwise", "pooled"}
    DEFAULT_CRPS_ALPHAS = np.arange(0.0025, 1.0, 0.0025)

    def __init__(
        self,
        name: str,
        experiment_seed: int,
        task: BaseTask,
        call_mode: str,
        wandb_enabled: bool = False,
        n_estimators: int = 8,
        device: str = "cpu",
        validation_metric: str = "rmse",
        crps_alphas: Optional[List[float]] = None,
        model_params: Optional[DictConfig] = None,
        plot_intermediate: bool = False,
    ):
        if call_mode not in self.VALID_CALL_MODES:
            raise ValueError(
                f"Unknown TabICL call mode: {call_mode}. "
                f"Expected one of {sorted(self.VALID_CALL_MODES)}."
            )
        if validation_metric not in {"rmse", "crps"}:
            raise ValueError("TabICLExperiment only supports validation_metric='rmse' or 'crps'.")

        super().__init__(name, experiment_seed, task, validation_metric, plot_intermediate=plot_intermediate)
        self.call_mode = call_mode
        self.wandb_enabled = wandb_enabled
        self.n_estimators = n_estimators
        self.tabicl_device = device
        self.crps_alphas = self._normalize_crps_alphas(crps_alphas)
        self.model_params = self._normalize_model_params(model_params)
        self.is_gaussian_prediction = False

    @staticmethod
    def _normalize_crps_alphas(crps_alphas: Optional[List[float]]) -> np.ndarray:
        alphas = TabICLExperiment.DEFAULT_CRPS_ALPHAS if crps_alphas is None else crps_alphas
        alphas = np.asarray(list(alphas), dtype=float)
        if alphas.ndim != 1 or len(alphas) == 0:
            raise ValueError("crps_alphas must be a non-empty one-dimensional list.")
        if np.any(alphas <= 0.0) or np.any(alphas >= 1.0):
            raise ValueError("crps_alphas must lie strictly between 0 and 1.")
        return np.unique(alphas)

    @staticmethod
    def _normalize_model_params(model_params: Optional[DictConfig]) -> Dict[str, Any]:
        if model_params is None:
            return {}
        if isinstance(model_params, DictConfig):
            return dict(OmegaConf.to_container(model_params, resolve=True))
        return dict(model_params)

    def fit(
        self,
        train_data: BaseDataset,
        validation_target_data: Optional[BaseDataset] = None,
        validation_context_data: Optional[BaseDataset] = None,
    ) -> "TabICLExperiment":
        self.train_data = train_data
        self.is_fitted = True
        return self

    def run(self, data_bundle: DataBundle) -> Dict[str, Dict[str, float]]:
        results = {}
        train_data = data_bundle.train
        validation_context, validation_target, test_context, test_target = self._resolve_splits(data_bundle)

        start_time = time.time()
        self.fit(train_data, validation_target_data=validation_target, validation_context_data=validation_context)
        fitting_time = time.time() - start_time
        self.logger.info(f"Fitting time: {fitting_time:.2f} seconds")

        self._set_prediction_seed(VALIDATION_SPLIT_NAME)
        start_time = time.time()
        validation_predictions = self._predict(
            validation_target,
            context_data=validation_context,
            random_state=self._prediction_seed(VALIDATION_SPLIT_NAME),
        )
        validation_prediction_time = time.time() - start_time
        self.logger.info(f"Validation prediction time: {validation_prediction_time:.2f} seconds")

        self._set_prediction_seed(TEST_SPLIT_NAME)
        start_time = time.time()
        test_predictions = self._predict(
            test_target,
            context_data=test_context,
            random_state=self._prediction_seed(TEST_SPLIT_NAME),
        )
        test_prediction_time = time.time() - start_time
        self.logger.info(f"Test prediction time: {test_prediction_time:.2f} seconds")

        if self.wandb_enabled:
            wandb.summary.update({
                "fitting_time_s": fitting_time,
                "validation_prediction_time_s": validation_prediction_time,
                "test_prediction_time_s": test_prediction_time,
            })

        self._set_evaluation_seed(VALIDATION_SPLIT_NAME)
        results[VALIDATION_SPLIT_NAME] = self.evaluate(validation_predictions, validation_target)
        self._set_evaluation_seed(TEST_SPLIT_NAME)
        results[TEST_SPLIT_NAME] = self.evaluate(test_predictions, test_target)
        return results

    def predict(
        self,
        target_data: BaseDataset,
        context_data: Optional[BaseDataset] = None,
        random_state: Optional[int] = None,
    ) -> TabICLPrediction:
        return self._predict(
            target_data,
            context_data=context_data,
            random_state=self.experiment_seed if random_state is None else random_state,
        )

    def _predict(
        self,
        target_data: BaseDataset,
        context_data: Optional[BaseDataset],
        random_state: int,
    ) -> TabICLPrediction:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        if context_data is None:
            raise ValueError("TabICL prediction requires context data.")

        if self.call_mode == "taskwise":
            return self._predict_taskwise(target_data, context_data, random_state=random_state)
        if self.task.name == FEW_SHOT_TASK_NAME:
            return self._predict_pooled_few_shot(target_data, context_data, random_state=random_state)
        return self._predict_pooled(target_data, context_data, random_state=random_state)

    def _predict_taskwise(
        self,
        target_data: BaseDataset,
        context_data: BaseDataset,
        random_state: int,
    ) -> TabICLPrediction:
        self.logger.info("Generating taskwise TabICL predictions.")
        target_group_ids = np.asarray(target_data.grouping_var).reshape(-1)
        context_group_ids = np.asarray(context_data.grouping_var).reshape(-1)
        unique_target_groups = pd.unique(target_group_ids)
        mean = np.full(len(target_data.response), np.nan, dtype=float)
        quantiles = np.full((len(target_data.response), len(self.crps_alphas)), np.nan, dtype=float)
        regressor = self._new_regressor(random_state)

        for group_index, group_id in enumerate(unique_target_groups, start=1):
            context_indices = np.flatnonzero(context_group_ids == group_id)
            target_indices = np.flatnonzero(target_group_ids == group_id)
            if len(context_indices) == 0:
                raise ValueError(f"No TabICL context rows found for group {group_id}.")

            regressor.fit(
                context_data.features.iloc[context_indices],
                context_data.response.iloc[context_indices].to_numpy(),
            )

            group_mean, group_quantiles = self._predict_with_quantiles(
                regressor,
                target_data.features.iloc[target_indices],
            )
            mean[target_indices] = group_mean
            quantiles[target_indices] = group_quantiles
            self._log_group_progress("taskwise", group_index, len(unique_target_groups), group_id)

        self._log_prediction_summary_by_group("taskwise", mean, target_group_ids)
        return TabICLPrediction(
            mean=mean,
            std=None,
            group_ids=target_group_ids,
            quantiles=quantiles,
            quantile_alphas=self.crps_alphas,
        )

    def _predict_pooled(
        self,
        target_data: BaseDataset,
        context_data: BaseDataset,
        random_state: int,
    ) -> TabICLPrediction:
        self.logger.info(
            f"Generating pooled TabICL predictions for {len(target_data.response)} "
            f"target rows across {len(pd.unique(target_data.grouping_var))} groups."
        )
        context_features, target_features = self._features_with_group_id(context_data, target_data)
        regressor = self._new_regressor(random_state)
        regressor.fit(context_features, context_data.response.to_numpy())
        mean, quantiles = self._predict_with_quantiles(regressor, target_features)
        self._log_prediction_summary_by_group(
            "pooled",
            mean,
            np.asarray(target_data.grouping_var).reshape(-1),
        )
        return TabICLPrediction(
            mean=mean,
            std=None,
            group_ids=np.asarray(target_data.grouping_var).reshape(-1),
            quantiles=quantiles,
            quantile_alphas=self.crps_alphas,
        )

    def _predict_pooled_few_shot(
        self,
        target_data: BaseDataset,
        support_data: BaseDataset,
        random_state: int,
    ) -> TabICLPrediction:
        self.logger.info("Generating few-shot pooled TabICL predictions.")
        target_group_ids = np.asarray(target_data.grouping_var).reshape(-1)
        support_group_ids = np.asarray(support_data.grouping_var).reshape(-1)
        unique_target_groups = pd.unique(target_group_ids)
        mean = np.full(len(target_data.response), np.nan, dtype=float)
        quantiles = np.full((len(target_data.response), len(self.crps_alphas)), np.nan, dtype=float)
        regressor = self._new_regressor(random_state)

        for group_index, group_id in enumerate(unique_target_groups, start=1):
            support_indices = np.flatnonzero(support_group_ids == group_id)
            target_indices = np.flatnonzero(target_group_ids == group_id)
            if len(support_indices) == 0:
                raise ValueError(f"No TabICL support rows found for group {group_id}.")

            group_support_data = BaseDataset(
                features=support_data.features.iloc[support_indices].reset_index(drop=True),
                response=support_data.response.iloc[support_indices].reset_index(drop=True),
                grouping_var=support_data.grouping_var.iloc[support_indices].reset_index(drop=True),
            )
            context_data = self._combine_context_data(self.train_data, group_support_data)
            group_target_data = BaseDataset(
                features=target_data.features.iloc[target_indices].reset_index(drop=True),
                response=target_data.response.iloc[target_indices].reset_index(drop=True),
                grouping_var=target_data.grouping_var.iloc[target_indices].reset_index(drop=True),
            )

            context_features, target_features = self._features_with_group_id(context_data, group_target_data)
            regressor.fit(context_features, context_data.response.to_numpy())
            group_mean, group_quantiles = self._predict_with_quantiles(regressor, target_features)
            mean[target_indices] = group_mean
            quantiles[target_indices] = group_quantiles
            self._log_group_progress("few-shot pooled", group_index, len(unique_target_groups), group_id)

        self._log_prediction_summary_by_group("few-shot pooled", mean, target_group_ids)
        return TabICLPrediction(
            mean=mean,
            std=None,
            group_ids=target_group_ids,
            quantiles=quantiles,
            quantile_alphas=self.crps_alphas,
        )

    def _log_group_progress(
        self,
        mode_name: str,
        completed_groups: int,
        total_groups: int,
        group_id: Any,
    ) -> None:
        self.logger.info(
            f"TabICL {mode_name} progress: {completed_groups}/{total_groups} "
            f"groups done (latest group: {group_id})."
        )

    def _log_prediction_summary_by_group(
        self,
        mode_name: str,
        mean: np.ndarray,
        group_ids: np.ndarray,
    ) -> None:
        group_means = []
        for group_id in pd.unique(group_ids):
            group_mask = group_ids == group_id
            group_pred = np.asarray(mean[group_mask], dtype=float)
            group_means.append(float(np.nanmean(group_pred)))

        if not group_means:
            return

        min_group_mean = float(np.nanmin(group_means))
        max_group_mean = float(np.nanmax(group_means))
        # For checking whether TabICL actually predicts different means across groups
        # Especially for pooled TabICL it sometimes looks as if all groups get the same predictions at the same target inputs.
        self.logger.info(
            f"TabICL {mode_name} prediction summary across groups: "
            f"group-mean range {min_group_mean:.6g} to {max_group_mean:.6g}."
        )

    @staticmethod
    def _combine_context_data(train_data: BaseDataset, support_data: BaseDataset) -> BaseDataset:
        """Append few-shot support rows to training rows for pooled TabICL calls."""
        return BaseDataset(
            features=pd.concat([train_data.features, support_data.features], ignore_index=True),
            response=pd.concat([train_data.response, support_data.response], ignore_index=True),
            grouping_var=pd.concat([train_data.grouping_var, support_data.grouping_var], ignore_index=True),
        )

    def _new_regressor(self, random_state: int) -> TabICLRegressor:
        params = dict(self.model_params)
        params.update({
            "n_estimators": self.n_estimators,
            "random_state": random_state,
        })
        if self.tabicl_device is not None:
            params["device"] = self.tabicl_device
        return TabICLRegressor(**params)

    def _features_with_group_id(
        self,
        context_data: BaseDataset,
        target_data: BaseDataset,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Add a shared categorical group-id feature to context and target rows."""
        context_features = context_data.features.copy()
        target_features = target_data.features.copy()
        all_groups = pd.concat(
            [context_data.grouping_var, target_data.grouping_var],
            ignore_index=True,
        )
        group_categories = pd.Index(pd.unique(all_groups))

        context_features[GROUPING_COLUMN_NAME] = pd.Categorical(
            context_data.grouping_var.to_numpy(),
            categories=group_categories,
        )
        target_features[GROUPING_COLUMN_NAME] = pd.Categorical(
            target_data.grouping_var.to_numpy(),
            categories=group_categories,
        )
        return context_features, target_features

    def _predict_with_quantiles(
        self,
        regressor: TabICLRegressor,
        target_features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return TabICL mean and quantile predictions with validated shapes."""
        mean = np.asarray(regressor.predict(target_features), dtype=float).reshape(-1)
        if len(mean) != len(target_features):
            raise ValueError(
                f"TabICL returned {len(mean)} mean predictions for {len(target_features)} target rows."
            )

        quantiles = regressor.predict(
            target_features,
            output_type="quantiles",
            alphas=self.crps_alphas.tolist(),
        )
        quantiles = np.asarray(quantiles, dtype=float)
        n_rows = len(target_features)
        n_quantiles = len(self.crps_alphas)

        if quantiles.ndim == 1:
            if n_rows == 1:
                quantiles = quantiles.reshape(1, -1)
            elif n_quantiles == 1:
                quantiles = quantiles.reshape(-1, 1)

        if quantiles.shape == (n_quantiles, n_rows) and quantiles.shape != (n_rows, n_quantiles):
            quantiles = quantiles.T

        if quantiles.shape != (n_rows, n_quantiles):
            raise ValueError(
                "Unexpected TabICL quantile shape. "
                f"Expected {(n_rows, n_quantiles)}, got {quantiles.shape}."
            )
        return mean, quantiles

    def evaluate(self, predictions: TabICLPrediction, target_data: BaseDataset) -> Dict[str, float]:
        y_true = target_data.response.to_numpy()
        mean = np.asarray(predictions.mean).reshape(-1)
        rmse = float(np.sqrt(np.mean((mean - y_true) ** 2)))
        crps = quantile_crps(
            y_true=y_true,
            quantiles=predictions.quantiles,
            alphas=predictions.quantile_alphas,
        )
        return {"crps": crps, "rmse": rmse}

    def evaluate_by_group(self, predictions: TabICLPrediction, target_data: BaseDataset) -> pd.DataFrame:
        """Evaluate TabICL predictions per group for diagnostic plots."""
        return groupwise_metrics(predictions, target_data)

    def save_model(self, model_folder: str, model_file_name: str):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")

        vars_path = Path(model_folder) / f"{model_file_name}_vars.pkl"
        vars_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "experiment_seed": self.experiment_seed,
            "name": self.name,
            "task": self.task,
            "call_mode": self.call_mode,
            "wandb_enabled": self.wandb_enabled,
            "n_estimators": self.n_estimators,
            "tabicl_device": self.tabicl_device,
            "validation_metric": self.validation_metric,
            "crps_alphas": self.crps_alphas,
            "model_params": self.model_params,
        }
        joblib.dump(state, vars_path)

    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "TabICLExperiment":
        vars_path = Path(model_folder) / f"{model_file_name}_vars.pkl"
        state = joblib.load(vars_path)

        instance = cls.__new__(cls)
        for key, value in state.items():
            setattr(instance, key, value)
        instance.is_fitted = True
        instance.is_gaussian_prediction = False
        instance.device = torch.device("cpu")
        instance.crps_samples = 20
        instance.plot_intermediate = False
        instance.logger = logging.getLogger(cls.__name__)
        return instance


class TabICLTaskwiseExperiment(TabICLExperiment):
    """TabICL_T class that calls TabICL for each group separately."""

    def __init__(
        self,
        name: str,
        experiment_seed: int,
        task: BaseTask,
        wandb_enabled: bool = False,
        n_estimators: int = 8,
        device: str = "cpu",
        validation_metric: str = "rmse",
        crps_alphas: Optional[List[float]] = None,
        model_params: Optional[DictConfig] = None,
        plot_intermediate: bool = False,
    ):
        super().__init__(
            name=name,
            experiment_seed=experiment_seed,
            task=task,
            call_mode="taskwise",
            wandb_enabled=wandb_enabled,
            n_estimators=n_estimators,
            device=device,
            validation_metric=validation_metric,
            crps_alphas=crps_alphas,
            model_params=model_params,
            plot_intermediate=plot_intermediate,
        )


class TabICLPooledExperiment(TabICLExperiment):
    """TabICL_P class for the pooled experiments, 
    where context points from many groups (including all the training observations) are provided
    """

    def __init__(
        self,
        name: str,
        experiment_seed: int,
        task: BaseTask,
        wandb_enabled: bool = False,
        n_estimators: int = 8,
        device: str = "cpu",
        validation_metric: str = "rmse",
        crps_alphas: Optional[List[float]] = None,
        model_params: Optional[DictConfig] = None,
        plot_intermediate: bool = False,
    ):
        super().__init__(
            name=name,
            experiment_seed=experiment_seed,
            task=task,
            call_mode="pooled",
            wandb_enabled=wandb_enabled,
            n_estimators=n_estimators,
            device=device,
            validation_metric=validation_metric,
            crps_alphas=crps_alphas,
            model_params=model_params,
            plot_intermediate=plot_intermediate,
        )
