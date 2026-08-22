"""NPBoost model implementation.

This module provides the NPBoost model, which integrates gradient boosting
(via LightGBM) for modeling fixed effects with Neural Processes for capturing
group-specific random effects. The model training alternates between:

1.  Training a Neural Process on the current residualized responses.
2.  Using the trained NP to compute custom gradients for the boosting algorithm.
3.  Updating the LightGBM model with a single boosting step.
"""

import copy
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import time
import numpy as np
import pandas as pd
import torch
from torch.distributions import Independent, Normal
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from lightgbm import Booster, Dataset, train as lgb_train
from lightgbm.callback import EarlyStopException as LightGBMEarlyStopException
from omegaconf import DictConfig

from src.data.datasets import (
    GradientSplitDataset,
    NPBDataset,
    PairedFunctionSampler,
    PairedNPBDataset,
    gradient_split_collate_fn,
    paired_collate_fn,
)
from src.evaluation.crps import crps_eval
from src.evaluation.model_predictions import GroupedModelPrediction
from src.models.neural_process import NeuralProcess
from src.task.base_task import BaseTask
from src.training.losses import NPMLLoss
from src.training.np_trainer import NPTrainer
from src.utils.helpers import set_seed

logger = logging.getLogger(__name__)

INTERMEDIATE_PLOTTING_ROUNDS = [1, 2, 5, 10, 25, 50, 100, 150, 250]


@contextmanager
def force_torch_cpu_load():
    """Temporarily force ``torch.load`` calls inside joblib loading onto CPU."""
    original_torch_load = torch.load

    def forced_cpu_load(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_torch_load(*args, **kwargs)

    torch.load = forced_cpu_load
    try:
        yield
    finally:
        torch.load = original_torch_load


class NPBoost:
    """A hybrid model combining gradient boosting and Neural Processes.

    This model is designed for grouped data, using:
    1. LightGBM to model population-level (fixed) effects.
    2. A Neural Process to model group-specific (random) effects.
    Training proceeds by iteratively alternating between the two components.

    Args:
        name: Name of the experiment/model.
        experiment_seed: Random seed for reproducibility.
        task: Prediction task (in-context, or few-shot).
        validation_metric: Metric used for early stopping ('rmse', 'crps', or 'fixed_effect_residual_rmse').
        device: Device for Neural Process training ('cpu' or 'cuda').
        crps_samples: Number of samples for CRPS evaluation.
        max_boosting_rounds: Maximum number of boosting rounds.
        min_delta: Minimum change in validation metric to qualify as improvement.
        patience: Number of rounds without improvement before early stopping.
        restore_best_model: Whether to restore the best model after early stopping.
        np_params: Configuration parameters for the Neural Process model.
        lgbm_params: Configuration parameters for the LightGBM model.
        np_train_params: Training parameters for the Neural Process (e.g., learning rate, batch size).
        fixed_effect_warm_start_rounds: Number of initial boosting rounds without NP updates (default 0).
        compute_aux_validation_metrics: Whether to compute auxiliary metrics during validation (default True).
        plot_intermediate: Whether to create plots for intermediate results during training (default False).
    """

    def __init__(
        self,
        name: str,
        experiment_seed: int,
        task: BaseTask,
        validation_metric: str,
        device: str,
        crps_samples: int,
        max_boosting_rounds: int,
        min_delta: float,
        patience: int,
        restore_best_model: bool,
        np_params: DictConfig,
        lgbm_params: DictConfig,
        np_train_params: DictConfig,
        fixed_effect_warm_start_rounds: int = 0,
        compute_aux_validation_metrics: bool = True,
        plot_intermediate: bool = False,
    ):
        self.name = name
        self.experiment_seed = experiment_seed
        self.task = task
        self.validation_metric = validation_metric
        self.device = torch.device(device)
        self.crps_samples = crps_samples
        self.max_boosting_rounds = max_boosting_rounds
        self.fixed_effect_warm_start_rounds = fixed_effect_warm_start_rounds
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_model = restore_best_model
        self.np_params = np_params
        self.lgbm_params = lgbm_params
        self.np_train_params = np_train_params
        self.compute_aux_validation_metrics = compute_aux_validation_metrics
        self.plot_intermediate = plot_intermediate

        set_seed(experiment_seed)
        self.is_fitted = False
        self.is_gaussian_prediction = False
        self.logger = logging.getLogger("NPBoost")
        self.sigma_e_raw_final = None
        self.sigma_e_raw_min = None
        self.sigma_e_raw_min_boosting_round = None
        self.min_sigma_e = None

    def _decoder_sigma_e_raw(self) -> float:
        """Return the NP decoder's learned global noise scale before the minimum clamp."""
        with torch.no_grad():
            return float(torch.exp(self.neural_process_.decoder.log_sigma_e).detach().cpu().item())

    def _decoder_min_sigma_e(self) -> float:
        """Return the decoder lower bound for the global noise scale."""
        return float(self.neural_process_.decoder.min_sigma_e)

    def _record_sigma_e_raw(self, step: int) -> float:
        """Update sigma_e diagnostics for one boosting round and return the current value."""
        sigma_e_raw = self._decoder_sigma_e_raw()
        self.sigma_e_raw_final = sigma_e_raw

        # The minimal value that Neural Process sigma_e can reach. (it is capped by the decoder's min_sigma_e value)
        self.min_sigma_e = float(self.neural_process_.decoder.min_sigma_e)

        if self.sigma_e_raw_min is None or sigma_e_raw < self.sigma_e_raw_min:
            self.sigma_e_raw_min = sigma_e_raw
            self.sigma_e_raw_min_boosting_round = step
        return sigma_e_raw

    def sigma_e_summary(self) -> Dict[str, float]:
        """Return a summary of the sigma_e diagnostics.
        
        This may be helpful to understand the behavior of the Neural Process decoder's global noise parameter during training.
        """
        return {
            "diagnostics/sigma_e_raw_final": self.sigma_e_raw_final, # The final value of the Neural Process decoder's sigma_e parameter after training.
            "diagnostics/sigma_e_raw_min": self.sigma_e_raw_min, # The minimum value of the Neural Process decoder's sigma_e parameter observed during training.
            "diagnostics/sigma_e_raw_min_epoch": self.sigma_e_raw_min_boosting_round, # The boosting round at which the minimum sigma_e value was observed.
            "diagnostics/min_sigma_e": self.min_sigma_e, # The minimum value that the Neural Process decoder's sigma_e parameter can reach theoretically.
        }

    def fit(
        self,
        train_npbd: NPBDataset,
        validation_target_npbd: Optional[NPBDataset] = None,
        validation_context_npbd: Optional[NPBDataset] = None,
        log_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        validation_callback: Optional[
            Callable[["NPBoost", NPBDataset, NPBDataset, GroupedModelPrediction, Dict[str, float]], None]
        ] = None,
    ) -> "NPBoost":
        """Fit NPBoost with alternating Neural Process and LightGBM updates.

        The training alternates between:
        1. Residualizing the response by the current fixed-effect prediction.
        2. Training the Neural Process on those residualized responses.
        3. Computing NP-based gradients and Hessians for LightGBM.
        4. Updating the fixed-effect booster by one boosting step.

        Args:
            train_npbd: Grouped training data. ``features`` feed LightGBM and
                ``np_features`` feed the Neural Process.
            validation_target_npbd: Validation target rows used for early
                stopping. If omitted, all boosting rounds are trained.
            validation_context_npbd: Validation context/support rows paired with
                ``validation_target_npbd``.
            log_callback: Optional callback for logging after every boosting round.
            validation_callback: Optional callback for additional functionality after validation (e.g. plotting predictions).

        Returns:
            The fitted model.
        """

        self.logger.info("Starting NPBoost training...")
        self._initialize_components(train_npbd)

        # Initialize diagnostic variables for tracking global noise scale parameter of Neural Process
        self.sigma_e_raw_final = None
        self.sigma_e_raw_min = None
        self.sigma_e_raw_min_boosting_round = None
        self.min_sigma_e = self._decoder_min_sigma_e()

        # Create deep copy to avoid modifying original data
        train_data_copy = copy.deepcopy(train_npbd)
        
        # Initialize training history for analysis and debugging
        self.grad_history = []  # Track gradients over boosting rounds
        self.hess_history = []  # Track Hessians over boosting rounds
        self.f_pred_history = []  # Track fixed effect predictions

        has_validation = validation_target_npbd is not None and validation_context_npbd is not None
        if has_validation:
            self.early_stopped = False
            self.val_best = float("inf")
            self.val_metric_best_boosting_round = 0
            self.best_model = None
            self.best_np_state = None
            self.patience_counter = 0
            self.n_boosting_rounds_trained = self.max_boosting_rounds
            self.val_hist = []
        else:
            self.early_stopped = False
            self.val_best = float("nan")
            self.val_metric_best_boosting_round = self.max_boosting_rounds
            self.n_boosting_rounds_trained = self.max_boosting_rounds

        if self.fixed_effect_warm_start_rounds < 0:
            raise ValueError("fixed_effect_warm_start_rounds must be non-negative.")
        if self.fixed_effect_warm_start_rounds >= self.max_boosting_rounds:
            raise ValueError("fixed_effect_warm_start_rounds must be smaller than max_boosting_rounds.")
        
        # Initialize fixed effect predictions as zeros (no prior knowledge)
        f_pred = np.zeros(len(train_data_copy.features))

        if self.fixed_effect_warm_start_rounds > 0:
            f_pred = self._run_fixed_effect_warm_start(
                train_npbd=train_npbd,
                train_data_copy=train_data_copy,
                validation_target_npbd=validation_target_npbd,
                log_callback=log_callback,
            )

        # Main boosting loop: alternate between NP training and LightGBM updates
        for round_idx in range(self.fixed_effect_warm_start_rounds, self.max_boosting_rounds):
            self.current_boosting_round = round_idx + 1
            self.logger.info(f"Boosting round {self.current_boosting_round}/{self.max_boosting_rounds}")

            # Step 1: Update NP dataset with residualized responses (targets = original_targets - fixed_effects)
            train_data_copy.update_responses(train_npbd.response - f_pred)
            
            # Step 2: Train Neural Process on current residualized responses
            loss = self.np_trainer.train(
                train_data=train_data_copy,
                batch_size=self.np_train_params.batch_size,
                max_epochs=self.np_train_params.epochs_per_round,
                wandb_enabled=False,
            )
            sigma_e_raw = self._record_sigma_e_raw(self.current_boosting_round)
            log_data = {
                "np_loss": loss,
                "boosting_round": self.current_boosting_round,
                "diagnostics/sigma_e_raw": sigma_e_raw,
            }

            # Step 3: Compute custom gradients using trained Neural Process
            self.logger.info("Computing gradients for LightGBM update...")
            # Switch Neural Process to evaluation mode for gradient computation
            with torch.no_grad():
                self.neural_process_.eval()
                custom_objective = self._get_loss_function(train_data_copy)

            # Step 4: Update LightGBM model using Neural Process-derived gradients
            grad, hess = custom_objective()
            self.grad_history.append(grad)
            self.hess_history.append(hess)

            # Perform one boosting step with custom gradients
            self.lgbm_booster_.update(fobj=custom_objective)
            # Clean up memory
            del custom_objective

            self.logger.info("Updated the tree ensemble.")
            train_fixed_effect_pred = self.lgbm_booster_.predict(train_data_copy.features)
            train_fixed_effect_residual_rmse = np.sqrt(np.mean((train_npbd.response - train_fixed_effect_pred) ** 2))
            log_data["train_fixed_effect_residual_rmse"] = train_fixed_effect_residual_rmse

            if validation_target_npbd is not None:
                val_fixed_effect_pred = self.lgbm_booster_.predict(validation_target_npbd.features)
                val_fixed_effect_residual_rmse = np.sqrt(
                    np.mean((validation_target_npbd.response - val_fixed_effect_pred) ** 2)
                )
                log_data["val_fixed_effect_residual_rmse"] = val_fixed_effect_residual_rmse
                self.logger.info(
                    f"Fixed effect residual RMSE: train: {train_fixed_effect_residual_rmse:.3f}, "
                    f"val: {val_fixed_effect_residual_rmse:.3f}"
                )

            f_pred = train_fixed_effect_pred
            self.f_pred_history.append(f_pred)

            if has_validation:
                self.logger.info("Validating model on validation data.")
                val_loss, validation_predictions, validation_results = self._validate_model(
                    validation_context_npbd,
                    validation_target_npbd,
                    self.crps_samples,
                )
                if validation_callback is not None:
                    validation_callback(
                        self,
                        validation_context_npbd,
                        validation_target_npbd,
                        validation_predictions,
                        validation_results,
                    )

                self.early_stopped = self._check_early_stopping(val_loss, round_idx)
                log_data.update({
                    "val_rmse": self.val_rmse_latest,
                    "val_crps": self.val_crps_latest,
                    "val_fixed_effect_residual_rmse": self.val_fixed_effect_residual_rmse_latest,
                    "current_val_metric": val_loss,
                })
                self.logger.info(f"Patience: {self.patience_counter}/{self.patience}.")
                self.logger.info(f"Best validation {self.validation_metric.upper()}: {self.val_best}.\n")

                if self.early_stopped:
                    self.logger.info(f"Stopped after {self.n_boosting_rounds_trained} boosting rounds.\n")
                    if log_callback is not None:
                        log_callback(self.current_boosting_round, log_data)
                    break

            if log_callback is not None:
                log_callback(self.current_boosting_round, log_data)

        self.logger.info("Finished NPBoost training!")
        if self.early_stopped:
            self.logger.info(f"Early stopped training after iteration {self.n_boosting_rounds_trained}.")
        self.logger.info(f"Best validation {self.validation_metric.upper()}: {self.val_best}.")

        if has_validation and self.restore_best_model and self.best_model is not None:
            # Restore the best model
            self.lgbm_booster_.model_from_string(self.best_model)
            if self.best_np_state is not None:
                self.neural_process_.load_state_dict(self.best_np_state)
            self.logger.info(f"Restored model with best validation {self.validation_metric.upper()}.")

        self.sigma_e_raw_final = self._decoder_sigma_e_raw()
        self.min_sigma_e = self._decoder_min_sigma_e()
        self.is_fitted = True
        return self

    def _initialize_components(self, train_npbd: NPBDataset) -> None:
        """Initialize the Neural Process trainer and LightGBM booster."""
        self.neural_process_ = NeuralProcess(
            **self.np_params,
            x_dim=train_npbd.np_features.shape[1],
        ).to(self.device)
        # Initialize Adam optimizer for Neural Process training
        self.np_optimizer = torch.optim.Adam(
            self.neural_process_.parameters(),
            lr=self.np_train_params.learning_rate,
        )
        if self.np_train_params.use_linear_lr_scheduler:
            # Set linear learning rate scheduler if specified in training parameters
            self.learning_rate_scheduler = LinearLR(
                self.np_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.np_train_params.epochs_per_round
                * max(1, self.max_boosting_rounds - self.fixed_effect_warm_start_rounds),
            )
        else:
            self.learning_rate_scheduler = None

        # Initialize Neural Process Marginal Likelihood loss function
        self.npml_loss = NPMLLoss()
        # Initialize the trainer object for the Neural Process part
        self.np_trainer = NPTrainer(
            model=self.neural_process_,
            optimizer=self.np_optimizer,
            loss=self.npml_loss,
            scheduler=self.learning_rate_scheduler,
            logger=self.logger,
            device=self.device,
            task=self.task,
            plot_intermediate=self.plot_intermediate,
        )
        # LightGBM still needs a Dataset object, even though training later uses
        # custom gradients and Hessians instead of the built-in objective.
        self.lgbm_booster_ = Booster(
            params=dict(self.lgbm_params),
            train_set=Dataset(data=train_npbd.features, label=train_npbd.response),
        )

    def _run_fixed_effect_warm_start(
        self,
        train_npbd: NPBDataset,
        train_data_copy: NPBDataset,
        validation_target_npbd: Optional[NPBDataset],
        log_callback: Optional[Callable[[int, Dict[str, float]], None]],
    ) -> np.ndarray:
        """Run plain boosting rounds before Neural Process updates start.

        This optional warm start is mainly an exploratory tool. It logs fixed
        effect residual RMSE, but does not run early stopping or intermediate
        validation plots during the warm-start rounds.
        """
        
        self.logger.info(
            f"Warm-starting fixed effect with {self.fixed_effect_warm_start_rounds} plain boosting rounds."
        )
        f_pred = np.zeros(len(train_data_copy.features))
        for warm_start_idx in range(self.fixed_effect_warm_start_rounds):
            self.current_boosting_round = warm_start_idx + 1
            self.lgbm_booster_.update()
            f_pred = self.lgbm_booster_.predict(train_data_copy.features)
            self.f_pred_history.append(f_pred)

            log_data = {
                "warm_start_round": self.current_boosting_round,
                "train_fixed_effect_residual_rmse": np.sqrt(np.mean((train_npbd.response - f_pred) ** 2)),
                "diagnostics/sigma_e_raw": self._record_sigma_e_raw(self.current_boosting_round),
            }
            if validation_target_npbd is not None:
                val_fixed_effect_pred = self.lgbm_booster_.predict(validation_target_npbd.features)
                log_data["val_fixed_effect_residual_rmse"] = np.sqrt(
                    np.mean((validation_target_npbd.response - val_fixed_effect_pred) ** 2)
                )

            self.logger.info(
                f"Warm-start fixed effect residual RMSE: "
                f"train: {log_data['train_fixed_effect_residual_rmse']:.3f}, "
                f"val: {log_data.get('val_fixed_effect_residual_rmse', float('nan')):.3f}"
            )
            if log_callback is not None:
                log_callback(self.current_boosting_round, log_data)

        return f_pred

    def _validate_model(
        self,
        validation_context: NPBDataset,
        validation_target: NPBDataset,
        crps_samples: int = 20,
    ) -> Tuple[float, GroupedModelPrediction, Dict[str, float]]:
        """
        Validate the current NPBoost model on the provided validation datasets.
        Returns the validation loss, predictions, and a dictionary of validation metrics.
        """
        
        validation_predictions = self.predict(
            target_npbd=validation_target,
            context_npbd=validation_context,
        )
        needs_plots = self.current_boosting_round in INTERMEDIATE_PLOTTING_ROUNDS and self.plot_intermediate
        # Optionally do not compute CRPS if not needed for early stopping or auxiliary metrics, to save computation time.
        needs_crps = self.validation_metric == "crps" or self.compute_aux_validation_metrics or needs_plots

        val_rmse = self._rmse_from_prediction_arrays(validation_predictions, validation_target)
        val_fixed_effect_residual_rmse = self._fixed_effect_residual_rmse_from_prediction_arrays(
            validation_predictions,
            validation_target,
        )
        val_crps = (
            crps_eval(validation_predictions, validation_target, crps_samples, device=self.device)
            if needs_crps
            else float("nan")
        )
        self.val_crps_latest = val_crps
        self.val_rmse_latest = val_rmse
        self.val_fixed_effect_residual_rmse_latest = val_fixed_effect_residual_rmse

        validation_results = {
            "crps": val_crps,
            "rmse": val_rmse,
            "fixed_effect_residual_rmse": val_fixed_effect_residual_rmse,
        }
        if self.validation_metric == "crps":
            val_loss = val_crps
        elif self.validation_metric == "rmse":
            val_loss = val_rmse
        elif self.validation_metric == "fixed_effect_residual_rmse":
            val_loss = val_fixed_effect_residual_rmse
        else:
            raise ValueError(f"Unknown validation metric: {self.validation_metric}")

        self.val_hist.append(val_loss)
        return val_loss, validation_predictions, validation_results

    @staticmethod
    def _rmse_from_prediction_arrays(predictions: GroupedModelPrediction, target_npbd: NPBDataset) -> float:
        """Compute RMSE after restoring grouped predictions to target-row order."""
        predictions_ordered = predictions.reorder_to_original()
        residuals = predictions_ordered.mean - target_npbd.response.to_numpy()
        return float(np.sqrt(np.mean(residuals ** 2)))

    @staticmethod
    def _fixed_effect_residual_rmse_from_prediction_arrays(
        predictions: GroupedModelPrediction,
        target_npbd: NPBDataset,
    ) -> float:
        """Compute RMSE of the LightGBM fixed effect against the target response."""
        predictions_ordered = predictions.reorder_to_original()
        residuals = target_npbd.response.to_numpy() - predictions_ordered.fixed_effect
        return float(np.sqrt(np.mean(residuals ** 2)))

    def _check_early_stopping(self, val_loss: float, round_idx: int) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            val_loss: Current validation loss
            round_idx: Current boosting round

        Returns:
            bool: True if fitting should stop, False otherwise
        """

        # Check for improvement in validation loss
        if val_loss < (self.val_best - self.min_delta):
            self.val_best = val_loss
            self.val_metric_best_boosting_round = round_idx + 1
            self.patience_counter = 0

            if self.restore_best_model:
                self.best_model = copy.deepcopy(self.lgbm_booster_.model_to_string())
                self.best_np_state = copy.deepcopy(self.neural_process_.state_dict())
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.n_boosting_rounds_trained = round_idx + 1
            return True

        return False

    def predict(self, target_npbd: NPBDataset, context_npbd: NPBDataset) -> GroupedModelPrediction:
        """
        Takes as input a target dataset and predicts the response using the trained NPBoost model.
        We do this in a batched manner, in order to speed it up.

        Args:
            target_npbd: NPBDataset containing the target features for prediction
            context_npbd: NPBDataset containing the context data (features and responses) for the Neural Process

        Returns:
            A grouped model prediction object containing the predicted means, and predictive distributions for each group in the target dataset.
        """
        context_data_copy = copy.deepcopy(context_npbd)

        fixed_effect_context = self.lgbm_booster_.predict(context_data_copy.features)
        fixed_effect_target = self.lgbm_booster_.predict(target_npbd.features)

        context_residuals = context_data_copy.response - pd.Series(fixed_effect_context)
        context_data_copy.update_responses(context_residuals)

        z_samples = self.neural_process_.n_z_samples_test
        all_means, all_stds, all_groups, all_original_indices, all_fixed_effects = [], [], [], [], []
        predictive_distributions = {}

        prediction_dataset = PairedNPBDataset(
            context_npbd=context_data_copy,
            target_npbd=target_npbd,
        )
        prediction_sampler = PairedFunctionSampler(
            dataset=prediction_dataset,
            batch_size=self.np_train_params.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            self.neural_process_.eval()
            for batch_indices in prediction_sampler:
                groups = [prediction_dataset.unique_groups[index] for index in batch_indices]
                batch = paired_collate_fn([prediction_dataset[index] for index in batch_indices])
                x_c, y_c, _, x_t, _, original_indices_batch = batch
                # Batched by group: x_c/y_c are (batch_size, n_context, x_dim/y_dim),
                # and x_t is (batch_size, n_target, x_dim).
                x_c = x_c.to(self.device)
                y_c = y_c.to(self.device)
                x_t = x_t.to(self.device)

                # Select the fixed effect for each group's target rows and keep y_dim.
                fixed_effect = np.stack(
                    [
                        fixed_effect_target[original_indices.cpu().numpy()]
                        for original_indices in original_indices_batch
                    ]
                )[..., np.newaxis]  # (batch_size, n_target, 1)

                # Add the z-sample dimension so it can be added to NP predictions.
                mean_delta = torch.tensor(
                    fixed_effect,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0).expand(z_samples, -1, -1, -1)  # (z_samples, batch_size, n_target, 1)

                # The Neural Process returns one predictive distribution per latent sample.
                np_prediction, _ = self.neural_process_.forward(x_c, y_c, x_t)
                shifted_mean = np_prediction.mean + mean_delta  # (z_samples, batch_size, n_target, 1)
                std = np_prediction.stddev  # (z_samples, batch_size, n_target, 1)

                mean_out = shifted_mean.mean(dim=0).squeeze(-1)  # (batch_size, n_target)
                mean_of_var = (std ** 2).mean(dim=0).squeeze(-1)  # (batch_size, n_target)
                var_of_mean = shifted_mean.var(dim=0).squeeze(-1)  # (batch_size, n_target)
                std_out = (mean_of_var + var_of_mean).sqrt()  # (batch_size, n_target)

                n_target = x_t.shape[1]
                for batch_idx, group in enumerate(groups):
                    # Restore the old per-group distribution shape by keeping a singleton batch dimension.
                    group_shifted_mean = shifted_mean[:, batch_idx : batch_idx + 1]  # (z_samples, 1, n_target, 1)
                    group_std = std[:, batch_idx : batch_idx + 1]  # (z_samples, 1, n_target, 1)
                    group_fixed_effect = fixed_effect[batch_idx]  # (n_target, 1)
                    original_indices = original_indices_batch[batch_idx]

                    predictive_distributions[group] = Independent(
                        Normal(loc=group_shifted_mean, scale=group_std),
                        1,
                    )
                    all_means.append(mean_out[batch_idx].cpu().numpy())
                    all_stds.append(std_out[batch_idx].cpu().numpy())
                    all_groups.extend([group] * n_target)
                    all_original_indices.extend(original_indices.cpu().numpy())
                    all_fixed_effects.extend(group_fixed_effect)

        all_means = np.concatenate(all_means)
        all_stds = np.concatenate(all_stds)
        all_groups = np.array(all_groups)
        all_original_indices = np.array(all_original_indices)
        all_fixed_effects = np.concatenate(all_fixed_effects)

        return GroupedModelPrediction(
            mean=all_means,
            std=all_stds,
            original_indices=all_original_indices,
            group_ids=all_groups,
            # Maps group IDs to their corresponding predictive distributions -> every feature in the group has z_samples predictive distributions (one for each sampled z).
            predictive_distributions=predictive_distributions,
            # The fixed effect predictions for each target point, which are the predictions from the LightGBM model.
            fixed_effect=all_fixed_effects,
            random_effect=all_means - all_fixed_effects,
        )

    def _predict(self, target_npbd: NPBDataset, context_npbd: NPBDataset) -> GroupedModelPrediction:
        """Compatibility wrapper used by experiment runners."""
        return self.predict(target_npbd=target_npbd, context_npbd=context_npbd)

    def _get_loss_function(self, train_data: NPBDataset) -> Callable:
        """Create LightGBM's custom objective for the current boosting round.

        Gradients are accumulated over task-specific context/target splits and
        then negated, because the derivation is for the log likelihood while
        LightGBM minimizes a loss.

        Args:
            train_data: Residualized grouped data for this boosting round.

        Returns:
            A callable returning the fixed gradient and Hessian arrays expected
            by LightGBM's ``update(fobj=...)`` API.
        """
        # Get the total number of points for the whole dataset
        n_total_dataset = len(train_data.features)

        # Initialise the result tensors (for all groups)
        grad_result = torch.zeros(n_total_dataset).to(self.device)
        hess_result = torch.zeros(n_total_dataset).to(self.device)

        gradient_dataset = GradientSplitDataset(train_data, self.task)
        gradient_sampler = PairedFunctionSampler(
            dataset=gradient_dataset,
            batch_size=self.np_train_params.batch_size,
            shuffle=False, # We do not need to shuffle the batches, but inside the batches, the context-target splits are randomly sampled.
        )
        gradient_loader = DataLoader(
            dataset=gradient_dataset,
            batch_sampler=gradient_sampler,
            collate_fn=gradient_split_collate_fn,
        )

        for _, x_c, y_c, x_t, y_t, original_indices, normalizer in gradient_loader:
            x_c = x_c.to(self.device)
            y_c = y_c.to(self.device)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)
            original_indices = original_indices.to(self.device)
            normalizer = normalizer.to(self.device).unsqueeze(-1)

            g, h = self._get_gradient(x_c, y_c, x_t, y_t)
            grad_result.index_put_(
                (original_indices.flatten(),),
                (g.squeeze(-1) / normalizer).flatten(),
                accumulate=True,
            )
            hess_result.index_put_(
                (original_indices.flatten(),),
                (h.squeeze(-1) / normalizer).flatten(),
                accumulate=True,
            )

        # LightGBM API expects the grad and hess for the negative log likelihood.
        # In our paper, we derive the gradient for the negative log likelihood, so we would not need to switch the sign.
        # Since here, it is still based on a previous implementation, we need to switch the sign (_get_gradient() returns gradient of log likelihood)
        grad_np = -grad_result.detach().cpu().numpy()
        hess_np = -hess_result.detach().cpu().numpy()

        # Define custom loss function for LightGBM
        def custom_loss(dtrain=None, preds=None):
            return grad_np, hess_np

        return custom_loss

    def _get_gradient(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return NP log-likelihood derivatives for one context/target batch.

        The derivatives are with respect to the current fixed-effect prediction
        at the target points. ``_get_loss_function`` handles the sign convention
        needed by LightGBM.

        Args:
            x_c: Context covariates
            y_c: Context response
            x_t: Target covariates
            y_t: Target response

        Returns:
            ``(grad, hess)`` tensors aligned with ``y_t``.
        """

        # Forward pass to get predictive distribution at target points
        with torch.no_grad():
            self.neural_process_.eval()
            predictive_dist, _ = self.neural_process_.forward(x_c, y_c, x_t)
        
        # Calculate the softmax weights for the weighting
        log_prob = predictive_dist.log_prob(y_t)
        weights = softmax(log_prob.sum(dim=-1), dim=0)

        # Reshape the weights so we can broadcast
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # Get parameters of predictive distribution
        mean = predictive_dist.mean
        variance = predictive_dist.variance

        # Calculate the normalised residuals
        delta = (y_t.unsqueeze(dim=0) - mean) / variance
        grad = torch.sum(delta * weights, dim=0)

        # Calculate the diagonal elements of the Hessian
        hess = torch.sum(
            (((delta - grad.unsqueeze(0)) * delta) - (1 / variance)) * weights,
            dim=0,
        )

        return grad, hess

    def save_model(self, model_folder: str, model_file_name: str):
        """Serialize the fitted NPBoost instance with joblib."""
        model_path = Path(f"{model_folder}/{model_file_name}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

    @classmethod
    def load_model(cls, model_folder: str, model_file_name: str) -> "NPBoost":
        """Load a saved NPBoost instance, falling back to CPU when CUDA is unavailable."""
        if not torch.cuda.is_available():
            with force_torch_cpu_load():
                instance = joblib.load(f"{model_folder}/{model_file_name}.pkl")

            if instance.device == torch.device("cuda"):
                instance.logger.warning(
                    "Model was trained on GPU but no GPU is available. Loading on CPU instead."
                )
                instance.device = torch.device("cpu")
                instance.np_trainer.device = instance.device
        else:
            instance = joblib.load(f"{model_folder}/{model_file_name}.pkl")
        return instance
