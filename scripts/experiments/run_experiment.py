#!/usr/bin/env python3
"""
This is the main entry point for running the experiments. It is designed to be flexible so that it can be used for both synthetic and real-data experiments.
It can also handle all the different types of models defined in configs/model (NPBoost, NP, GPLinear, ...) and tasks (prediction scenarios) (InContext, FewShot).
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Standard library imports
import logging
from pathlib import Path

# Third-party imports
import hydra
from hydra.utils import instantiate
import wandb
from omegaconf import DictConfig, OmegaConf

# Local imports
from src.constants import VALIDATION_SPLIT_NAME
from src.data.datasets import BundleFactory
from src.utils.helpers import set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # Setup experiment logger.
    log = logging.getLogger(cfg.data.experiment.name)

    experiment_seed = cfg.data.experiment.seed
    split_data_seed = cfg.data.experiment.split_data_seed
    plot_intermediate = (split_data_seed == 0)

    log.info("Starting experiment.")

    
    # Set random seeds for reproducibility
    set_seed(experiment_seed)

    data_provider = instantiate(cfg.data.provider)
    task = instantiate(cfg.task)
    model = instantiate(cfg.model, experiment_seed=experiment_seed, task=task, plot_intermediate=plot_intermediate)


    if cfg.data.type == "synthetic":
        split_df = data_provider.fetch_split_df(
            task=task,
            seed=split_data_seed)
        wandb_run_name = f"{model.name}-{task.name}-{cfg.data.synthetic.fixed_effect.name}-seed{split_data_seed}"
    else:
        split_df = data_provider.fetch_split_df(
            task=task,
            response_column=cfg.data.experiment.response_name,
            seed=split_data_seed)
        wandb_run_name = f"{model.name}-{task.name}-{cfg.data.experiment.response_name}-seed{split_data_seed}"

    
     # Initialize wandb run for experiment tracking
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=wandb_run_name,
            config=OmegaConf.to_container(cfg),
        )

    data_bundle = BundleFactory.create_bundle(split_df, task)
    
    results = model.run(data_bundle)

    log.info(f"Experiment completed. Final results: {results}")

    # Log final results to wandb if enabled
    if cfg.wandb.enabled:
        wandb.summary.update(results)
        wandb.finish()

    val_result = results[VALIDATION_SPLIT_NAME]
    if cfg.model.validation_metric == "rmse":
        log.info(f"Validation RMSE: {val_result['rmse']}")
    elif cfg.model.validation_metric == "crps":
        log.info(f"Validation CRPS: {val_result['crps']}")
    elif cfg.model.validation_metric == "fixed_effect_residual_rmse":
        log.info(f"Validation fixed-effect residual RMSE: {val_result['fixed_effect_residual_rmse']}")
    
    run_validation_result = val_result[cfg.model.validation_metric]

    return run_validation_result

if __name__ == "__main__":
    main()
