"""
Trainer class for Neural Process models with optional early stopping.

Provides training and prediction functionality for Neural Process variants,
handling loss computation, optimization, and performance monitoring.
"""

# Standard library imports
import copy
from logging import Logger
from typing import Any, Dict, Optional

# Third-party imports
from src.evaluation.model_predictions import GroupedModelPrediction, ModelPrediction
from src.evaluation.squared_error import point_pred_eval, squared_error
import wandb
import torch
from torch.distributions import Independent, Normal
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.func import vmap
from tqdm import tqdm
import numpy as np
from src.data.datasets import DataBundle
from src.plot.model_diagnostics import plot_residuals_vs_predicted_single, plot_single_model_coverage, plot_single_model_predictions
from src.data.data_layout import PROJECT_ROOT
from pathlib import Path
import matplotlib.pyplot as plt

from src.constants import FEW_SHOT_TASK_NAME, IN_CONTEXT_TASK_NAME, SUPPORT_ROLE_NAME, TARGET_ROLE_NAME
from src.task.base_task import BaseTask

# Local imports
from ..models.neural_process import BaseNP
from ..data.datasets import NPBDataset, FunctionSampler, PairedFunctionSampler, PairedNPBDataset, paired_collate_fn
from ..evaluation.crps import crps_eval, crps_score
from ..utils.helpers import split_context

INTERMEDIATE_PLOTTING_ROUNDS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 3000]

class NPTrainer:
    """
    Trainer class for Neural Process models.

    Handles training with optional early stopping, validation monitoring, and
    prediction with trained models. Supports integration with Weights & Biases
    for experiment tracking.
    """

    def __init__(
        self,
        model: BaseNP,
        optimizer: Optimizer,
        loss: Module,
        scheduler: StepLR = None,
        logger: Logger = None,
        device: str = "cuda",
        task: BaseTask = None,
        plot_intermediate: bool = False,
    ) -> None:
        """
        Initialize Neural Process trainer.

        Args:
            model: Neural Process model to train
            optimizer: PyTorch optimizer to be used for training
            loss: Loss function to be optimized
            scheduler: Learning rate scheduler
            logger: Logger for the subprocesses
            device: Device to run training on
            task: Task configuration (in-context, few-shot), for handling context/target splits during training
            plot_intermediate: Whether to plot at epochs defined in INTERMEDIATE_PLOTTING_ROUNDS (default: False)
        """

        # Initialize the trainer attributes
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        self.task = task
        self.plot_intermediate = plot_intermediate

        # Move model to device
        self.model.to(device)
        self.logger.info(f"Initialized NPTrainer with model on device: {device}")

        # Initialize loss tracking
        self.train_loss_hist = []

        # Track the global noise scale parameter for diagnostics
        self.sigma_e_raw_final = None
        self.sigma_e_raw_min = None
        self.sigma_e_raw_min_epoch = None
        self.min_sigma_e = None

    def _decoder_sigma_e_raw(self) -> float:
        """Return the decoder's learned global noise scale."""
        with torch.no_grad():
            return float(torch.exp(self.model.decoder.log_sigma_e).detach().cpu().item())

    def _decoder_min_sigma_e(self) -> float:
        """Return the lower bound used by the decoder for its global noise scale."""
        return float(self.model.decoder.min_sigma_e)

    def _record_sigma_e_raw(self, epoch: int) -> float:
        """Update sigma_e diagnostics for one epoch and return the current value."""
        sigma_e_raw = self._decoder_sigma_e_raw()
        self.sigma_e_raw_final = sigma_e_raw
        self.min_sigma_e = float(self.model.decoder.min_sigma_e)
        if self.sigma_e_raw_min is None or sigma_e_raw < self.sigma_e_raw_min:
            self.sigma_e_raw_min = sigma_e_raw
            self.sigma_e_raw_min_epoch = epoch
        return sigma_e_raw

    def _sigma_e_summary(self) -> Dict[str, float]:
        """Return sigma_e diagnostics for W&B."""
        return {
            "diagnostics/sigma_e_raw_final": self.sigma_e_raw_final,
            "diagnostics/sigma_e_raw_min": self.sigma_e_raw_min,
            "diagnostics/sigma_e_raw_min_epoch": self.sigma_e_raw_min_epoch,
            "diagnostics/min_sigma_e": self.min_sigma_e,
        }

    def _wandb_plot_key(self, plot_name: str, plot_kind: str, step_value: int) -> str:
        """Build a W&B key for intermediate diagnostic plots."""
        return f"plots/{plot_name}/{plot_kind}/{step_value:04d}"

    def _wandb_image(self, plot, caption: str):
        """Render plot as a W&B image and close the figure."""
        figure = plot.draw()
        try:
            return wandb.Image(figure, caption=caption)
        finally:
            plt.close(figure)

    def _log_wandb_media(self, media: Dict[str, Any], step: Optional[int] = None):
        """Log W&B media when it is enabled."""
        if self.wandb_enabled and media:
            wandb.log(media, step=step)
        

    def train(
        self,
        train_data: NPBDataset,
        batch_size: int = 16,
        max_epochs: int = 100,
        wandb_enabled: bool = True,
        validation_metric: str = "crps",
        validation_context: NPBDataset = None,
        validation_target: NPBDataset = None,
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best_model: bool = True,
        crps_samples: int = 20,
    ):
        """
        Train the Neural Process model with optional early stopping.

        Args:
            train_data: Training dataset containing grouped function observations
            batch_size: Number of functions per batch
            max_epochs: Maximum number of training epochs
            wandb_enabled: Whether to log metrics to Weights & Biases
            validation_metric: Validation metric to monitor for early stopping ("rmse" or "crps")
            validation_context: Validation context dataset for early stopping
            validation_target: Validation target dataset for early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in validation loss to qualify as improvement
            restore_best_model: Whether to restore best model weights after training
            crps_samples: Number of samples to be calculated to estimate validation CRPS

        Returns:
            Average training loss over final epoch (average log likelihood over all functions)
        """

        # Create data loader with function wise sampling for batch wise training
        ts = FunctionSampler(dataset=train_data, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(dataset=train_data, batch_sampler=ts)

        self.wandb_enabled = wandb_enabled

        self.batch_size = batch_size

        # Clear previous training history
        self.train_loss_hist = []
        self.sigma_e_raw_final = None
        self.sigma_e_raw_min = None
        self.sigma_e_raw_min_epoch = None
        self.min_sigma_e = float(self.model.decoder.min_sigma_e)

        if validation_target is not None:
            # Initialize variables for early stopping / validation. 

            # Early stopping state
            self.val_loss_best = float("inf")
            self.val_loss_best_epochs = -1
            self.best_model_state = None
            self.patience_counter = 0
            self.epochs_trained = max_epochs # Will be updated if early stopping is triggered

            # Early stopping parameters
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_model = restore_best_model
            self.val_loss_hist = []

            # Create validation data loader
            val_dataset = PairedNPBDataset(context_npbd=validation_context, target_npbd=validation_target)
            context_target_sampler = PairedFunctionSampler(dataset=val_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, 
                                    batch_sampler=context_target_sampler,
                                    collate_fn=paired_collate_fn)

        # Logging if logger is available
        if self.logger is not None:

            # Log the start of the Neural Process training indicating the maximum number of epochs
            self.logger.info(f"Starting NP training for {max_epochs} epochs...")

            # If validation data is provided, we also log information on early stopping
            if validation_target is not None:
                self.logger.info(f"Early stopping if validation {validation_metric.upper()} does not decrease for {self.patience} epochs.")

        # Main training loop
        with tqdm(range(max_epochs)) as pbar:
            for epoch in pbar:
                epoch_num = epoch + 1

                # Execute one training epoch and report average loss for that epoch
                avg_train_loss = self._train_one_epoch(train_loader)
                postfix = {"avg_train_loss": f"{avg_train_loss:.6f}"}

                # Validation already records sigma_e; without validation we log it once at the end of the epoch.
                sigma_logged_this_epoch = False

                # Add average training loss to wandb dictionary to be logged
                if wandb_enabled:
                    wandb_dict = {"avg_train_loss": avg_train_loss}

                # Validate if validation data is provided
                if validation_target is not None:

                    # Evaluate the model on the validation data
                    validation_metrics = self._validate_model(val_loader, crps_samples)
                    avg_val_loss = validation_metrics[validation_metric]

                    self.val_loss_hist.append(avg_val_loss)
                    postfix[f"current_val_{validation_metric}"] = f"{avg_val_loss:.6f}"

                    # Log metric to Weights & Biases if enabled
                    if wandb_enabled:
                        wandb_dict.update({
                            "val_rmse": validation_metrics["rmse"],
                            "val_crps": validation_metrics["crps"],
                            f"current_val_metric": avg_val_loss,
                        })

                    self.early_stopped = self._check_early_stopping(avg_val_loss, epoch)
                    sigma_e_raw = self._record_sigma_e_raw(epoch_num)
                    sigma_logged_this_epoch = True
                    postfix["sigma_e_raw"] = f"{sigma_e_raw:.6f}"
                    if wandb_enabled:
                        wandb_dict["diagnostics/sigma_e_raw"] = sigma_e_raw

                    # Check early stopping condition
                    if self.early_stopped:
                        postfix.update(
                            {
                                "patience": f"{self.patience_counter}/{self.patience}",
                                "status": f"STOPPED: Best validation {validation_metric.upper()}: {self.val_loss_best:.6f}",
                            }
                        )
                        pbar.set_postfix(postfix)

                        if wandb_enabled:
                            wandb.log(wandb_dict, step=epoch + 1)

                        break

                    else:
                        # Update progress bar with current validation metrics
                        postfix.update(
                            {
                                "patience": f"{self.patience_counter}/{self.patience}",
                                f"best_val_{validation_metric}": f"{self.val_loss_best:.6f}",
                            }
                        )

                        # Log metrics to Weights & Biases if enabled
                        if wandb_enabled:
                            wandb_dict.update({f"val_{validation_metric}_best": self.val_loss_best,
                                               f"val_{validation_metric}_best_epoch": self.val_loss_best_epochs})

                    if epoch_num in INTERMEDIATE_PLOTTING_ROUNDS and self.plot_intermediate:
                        # Create diagnostic plots for the current epoch and log them to Weights & Biases
                        model_name = "anp" if self.model.use_attention else "np"
                        # Generate predictions for the validation set
                        # This is not yet ideal, since this generates another set of predictions than in self._validate_model.
                        validation_predictions = self.predict(target_npbd=validation_target, context_npbd=validation_context)
                        
                        wandb_plots = {}

                        # Coverage plot
                        coverage_ggplot = plot_single_model_coverage(
                            model_name=model_name,
                            predictions=validation_predictions,
                            prediction_metrics=validation_metrics,
                            target_data=validation_target,
                            save_path=None,
                            dpi=200,
                            is_gaussian_prediction=False,
                        )
                        
                        wandb_plots[self._wandb_plot_key(model_name, "coverage", epoch_num)] = self._wandb_image(
                                coverage_ggplot,
                                caption=f"Coverage calibration for epoch {epoch_num}",
                            )
                        

                        if validation_target.features.shape[1] == 1:
                            # Plot predictions only if the input dimension is 1D    
                            pred_ggplot = plot_single_model_predictions(
                                model_name=model_name,
                                predictions=validation_predictions,
                                prediction_metrics=validation_metrics,
                                target_data=validation_target,
                                context_data=validation_context,
                                task=self.task,
                                n_groups=8,
                                dpi=200,
                                eval_samples=crps_samples,
                                is_gaussian_prediction=False,
                            )

                            wandb_plots[self._wandb_plot_key(model_name, "predictions", epoch_num)] = self._wandb_image(
                                pred_ggplot,
                                caption=f"Predictions and uncertainty for epoch {epoch_num}",
                            )
                        self._log_wandb_media(wandb_plots, step=epoch_num)

                if not sigma_logged_this_epoch:
                    sigma_e_raw = self._record_sigma_e_raw(epoch_num)
                    postfix["sigma_e_raw"] = f"{sigma_e_raw:.6f}"
                    if wandb_enabled:
                        wandb_dict["diagnostics/sigma_e_raw"] = sigma_e_raw

                # Set postfix
                pbar.set_postfix(postfix)

                # Log to Weights & Biases if enabled
                if wandb_enabled:
                    wandb.log(wandb_dict, step=epoch + 1)

                # If scheduler is on (in our case for NPBoost application, then perform step after each epoch)
                if self.scheduler is not None:
                    self.scheduler.step()

                

         # Log metrics to Weights & Biases if enabled
        if wandb_enabled:
            wandb.summary.update({
                # This is the best achieved validation metric, but the final validation performance will be calculated again, 
                # so it can be different from the final validation metric.
                f"val_{validation_metric}_best": self.val_loss_best,
                f"val_{validation_metric}_best_epoch": self.val_loss_best_epochs,
                f"val_{validation_metric}_best_restored": self.restore_best_model,
                "best_epoch": self.val_loss_best_epochs,
                "early_stopped": self.early_stopped,
                # This is the epoch at which the training was stopped, 
                # not necessarily the epoch restored as the best model (except if the best performance was achieved at epoch ``max_epochs``).
                "epochs_trained": self.epochs_trained,
            })


        # Logging if logger is available
        if self.logger is not None:

            # Log that training finished
            self.logger.info("Finished Neural Process Training!")

            # If early stopping was triggered inform user in which epoch
            if validation_target is not None:

                # Check if early stopping was triggered
                if self.early_stopped:
                    self.logger.info(
                        f"Early stopped training after {self.epochs_trained} epochs."
                    )

                # Report the Best {validation_metric} in any case.
                self.logger.info(f"Best validation {validation_metric.upper()}: {self.val_loss_best}.")

        # Restore best model weights if early stopping was used and restore_best_model is True
        if (
            validation_target is not None
            and self.restore_best_model
            and self.best_model_state is not None
        ):
            # Restore the model with the best validation loss
            self.model.load_state_dict(self.best_model_state)

            # If logger was provided log that best model was restored.
            if self.logger is not None:
                self.logger.info(f"Restored model with best validation {validation_metric.upper()}.")
                self.logger.info(f"Best validation {validation_metric.upper()} achieved at epoch {self.val_loss_best_epochs}.")

        self.sigma_e_raw_final = self._decoder_sigma_e_raw()
        self.min_sigma_e = self._decoder_min_sigma_e()
        if wandb_enabled:
            wandb.summary.update(self._sigma_e_summary())

        # Return avg training loss over last epoch
        return avg_train_loss

    def _train_one_epoch(self, train_loader):
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader containing training batches

        Returns:
            Average training loss for the epoch (average log likelihood over all functions)
        """

        # Put model into training mode
        self.model.train()

        # Reset the training statistics
        running_loss = 0.0
        total_groups = 0  # Number of functions (groups) in our dataset

        # Use gradient accumulation since batches are often small (only groups of the same size can be batched together).
        # We accumulate gradients over multiple batches until at least ``n_groups_before_gradient_update`` groups have been processed.
        # Then we perform an optimizer step.
        n_groups_before_gradient_update = self.batch_size
        accumulated_groups = 0

        # Process each batch of functions
        for batch in train_loader:

            # Unpack the data
            x, y, _ = batch
            x, y = x.to(self.device), y.to(self.device)

            # Split each function into context and target points
            function_split = split_context(x, y, context_size=self.task.training_context_size)
            x_context, y_context, x_target, y_target = function_split

            # Extract the number of groups in this batch
            batch_n_groups = x_target.shape[0]

            # Predict target distribution given context
            p_y, _ = self.model.forward(x_context, y_context, x_target)

            # Compute loss between predictions and targets. Here, self.loss() returns the NPML objective
            # for each of the functions in the batch (note that the NPML objective is not normalised for the number
            # of target points in each function). Take the average over functions in the batch to calulate loss.
            loss = self.loss(p_y, y_target).mean()
            
            # Weight the accumulated loss by the number of groups in the batch. Note that
            # every group does *not* contribute equally to the gradient, because the NPML
            # loss itself depends on the number of target points in each group.
            # This reflects our model evaluation setup, since we evaluate the average RMSE / CRPS over all points in the dataset. 
            # Groups with more target points contribute more to the evaluation metric.
            weighted_accumulation_loss = loss * (batch_n_groups / n_groups_before_gradient_update)
            weighted_accumulation_loss.backward()
            
            accumulated_groups += batch_n_groups

            if accumulated_groups >= n_groups_before_gradient_update:
                # Clip the gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Perform optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Reset accumulated groups
                accumulated_groups = 0

            # Accumulate loss for epoch average
            running_loss += loss.detach() * batch_n_groups
            total_groups += batch_n_groups

        if accumulated_groups > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Calculate average training loss for epoch
        avg_train_loss = (running_loss / total_groups).item()
        self.train_loss_hist.append(avg_train_loss)

        # Return average loss for this epoch
        return avg_train_loss

    def _validate_model(self, val_loader, crps_samples):
        """
        Validate on paired context/target batches.

        Args:
            val_loader: DataLoader yielding paired context and target groups.
            crps_samples: Number of samples used to estimate CRPS.

        Returns:
            Dictionary with pointwise validation metrics: ``"crps"`` and ``"rmse"``.
        """

        # Put model into validation mode
        self.model.eval()

        # Reset the training statistics
        epoch_rmse_loss = 0.0
        epoch_crps_loss = 0.0
        total_samples = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            for batch in val_loader:
                x_context, y_context, _, x_target, y_target, _ = batch
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)
                y_target = y_target.to(self.device)

                # Extract the number of training examples in that batch
                batch_samples = x_target.shape[0]  # number of functions in the batch
                function_samples = x_target.shape[1]  # n_target in each function

                # Forward pass without gradient computation
                p_y, _ = self.model.forward(x_context, y_context, x_target)

                # Sample from p_y. Size is [batch_size, n_z_samples, batch_size, n_targets, y_dim]
                y_sample = p_y.sample([crps_samples])

                # Bring into shape expected for batched crps score calculation
                y_sample = (
                    y_sample.flatten(0, 1)  # Flatten sample and latent sample dim
                    .squeeze(-1)  # One dimensional response
                    .transpose(0, 1)  # Batch to the left most dimension
                    .to(self.device)  # Move to specified device
                ) # [batch_size, n_z_samples * crps_samples, n_targets]

                # Bring y_target into the correct shape for crps score calculation
                y_target_squeezed = y_target.squeeze(-1) # [batch_size, n_targets]

                point_preds = p_y.mean.mean(dim=0).squeeze(-1) # [batch_size, n_targets]
                
                # Calculate batched RMSE and CRPS scores for the batch of functions
                batched_rmse_score = vmap(squared_error)
                batch_rmse = batched_rmse_score(point_preds, y_target_squeezed) # Here it holds the squared error
                batched_crps_score = vmap(crps_score)
                batch_crps = batched_crps_score(y_sample, y_target_squeezed)
    
                # Accumulate validation crps over all target samples and batch functions
                epoch_rmse_loss += batch_rmse.sum(-1).sum(0).item()
                epoch_crps_loss += batch_crps.sum(-1).sum(0).item()
                total_samples += batch_samples * function_samples
        avg_crps_loss = epoch_crps_loss / total_samples
        avg_rmse_loss = epoch_rmse_loss / total_samples
        avg_rmse_loss = np.sqrt(avg_rmse_loss) # Now we take the square root to report RMSE

        return {
            "crps": avg_crps_loss,
            "rmse": avg_rmse_loss,
        }

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """

        # Check for improvement in validation loss
        if val_loss < (self.val_loss_best - self.min_delta):

            # New best validation loss found
            self.val_loss_best = val_loss
            self.val_loss_best_epochs = epoch + 1
            self.patience_counter = 0

            # Save model state for potential restoration
            if self.restore_best_model:
                self.best_model_state = copy.deepcopy(self.model.state_dict())

        else:
            # No improvement, increment patience counter
            self.patience_counter += 1

        # Stop training if patience limit exceeded
        if self.patience_counter >= self.patience:
            self.epochs_trained = epoch + 1
            return True

        return False

    def predict(self, target_npbd: NPBDataset, context_npbd: NPBDataset):
        """
        Make batched predictions using the trained model.

        Args:
            target_npbd: NPBDataset for which predictions should be generated.
            context_npbd: NPBDataset containing the context data.

        Returns:
            A grouped model prediction object containing the predicted means, and predictive distributions for each group in the target dataset.
        """

        self.model.eval()

        all_means, all_stds, all_original_indices, all_groups = [], [], [], []
        predictive_distributions = {}

        prediction_dataset = PairedNPBDataset(
            context_npbd=context_npbd,
            target_npbd=target_npbd,
        )
        prediction_sampler = PairedFunctionSampler(
            dataset=prediction_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for batch_indices in prediction_sampler:
                groups = [prediction_dataset.unique_groups[index] for index in batch_indices]
                batch = paired_collate_fn([prediction_dataset[index] for index in batch_indices])
                x_context, y_context, _, x_target, _, original_indices_batch = batch
                # Batched by group: x_context/y_context are (batch_size, n_context, x_dim/y_dim),
                # and x_target is (batch_size, n_target, x_dim).
                x_context = x_context.to(self.device)
                y_context = y_context.to(self.device)
                x_target = x_target.to(self.device)

                # The Neural Process returns one predictive distribution per latent sample.
                np_prediction, _ = self.model.forward(x_context, y_context, x_target)
                mean = np_prediction.mean  # (z_samples, batch_size, n_target, 1)
                std = np_prediction.stddev  # (z_samples, batch_size, n_target, 1)

                mean_out = mean.mean(dim=0).squeeze(-1)  # (batch_size, n_target)
                mean_of_var = (std ** 2).mean(dim=0).squeeze(-1)  # (batch_size, n_target)
                var_of_mean = mean.var(dim=0).squeeze(-1)  # (batch_size, n_target)
                std_out = (mean_of_var + var_of_mean).sqrt()  # (batch_size, n_target)

                n_target = x_target.shape[1]
                for batch_idx, group_id in enumerate(groups):
                    # Restore the old per-group distribution shape by keeping a singleton batch dimension.
                    group_mean = mean[:, batch_idx : batch_idx + 1]  # (z_samples, 1, n_target, 1)
                    group_std = std[:, batch_idx : batch_idx + 1]  # (z_samples, 1, n_target, 1)
                    original_indices = original_indices_batch[batch_idx]

                    predictive_distributions[group_id] = Independent(
                        Normal(loc=group_mean, scale=group_std),
                        1,
                    )
                    all_means.append(mean_out[batch_idx].cpu().numpy())
                    all_stds.append(std_out[batch_idx].cpu().numpy())
                    all_groups.extend([group_id] * n_target)
                    all_original_indices.extend(original_indices.cpu().numpy())

        return GroupedModelPrediction(
            mean=np.concatenate(all_means),
            std=np.concatenate(all_stds),
            original_indices=np.array(all_original_indices),
            group_ids=np.array(all_groups),
            predictive_distributions=predictive_distributions,
        )
